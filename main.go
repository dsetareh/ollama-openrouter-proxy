package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	openai "github.com/sashabaranov/go-openai"
)

var modelFilter map[string]struct{}

func loadModelFilter(path string) (map[string]struct{}, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	filter := make(map[string]struct{})

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line != "" {
			filter[line] = struct{}{}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return filter, nil
}

func main() {
	r := gin.Default()
	// Load the API key from environment variables or command-line arguments.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		if len(os.Args) > 1 {
			apiKey = os.Args[1]
		} else {
			slog.Error("OPENAI_API_KEY environment variable or command-line argument not set.")
			return
		}
	}

	provider := NewOpenrouterProvider(apiKey)

	filter, err := loadModelFilter("models-filter")
	if err != nil {
		if os.IsNotExist(err) {
			slog.Info("models-filter file not found. Skipping model filtering.")
			modelFilter = make(map[string]struct{})
		} else {
			slog.Error("Error loading models filter", "Error", err)
			return
		}
	} else {
		modelFilter = filter
		slog.Info("Loaded models from filter:")
		for model := range modelFilter {
			slog.Info(" - " + model)
		}
	}

	r.GET("/", func(c *gin.Context) {
		c.String(http.StatusOK, "Ollama is running")
	})
	r.HEAD("/", func(c *gin.Context) {
		c.String(http.StatusOK, "")
	})

	r.GET("/api/tags", func(c *gin.Context) {
		models, err := provider.GetModels()
		if err != nil {
			slog.Error("Error getting models", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		filter := modelFilter
		// Construct a new array of model objects with extra fields
		newModels := make([]map[string]interface{}, 0, len(models))
		for _, m := range models {
			// Если фильтр пустой, значит пропускаем проверку и берём все модели
			if len(filter) > 0 {
				if _, ok := filter[m.Model]; !ok {
					continue
				}
			}
			newModels = append(newModels, map[string]interface{}{
				"name":        m.Name,
				"model":       m.Model,
				"modified_at": m.ModifiedAt,
				"size":        270898672,
				"digest":      "9077fe9d2ae1a4a41a868836b56b8163731a8fe16621397028c2c76f838c6907",
				"details":     m.Details,
			})
		}

		c.JSON(http.StatusOK, gin.H{"models": newModels})
	})

	r.POST("/api/show", func(c *gin.Context) {
		var request map[string]string
		if err := c.BindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}

		modelName := request["name"]
		if modelName == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Model name is required"})
			return
		}

		details, err := provider.GetModelDetails(modelName)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}

		c.JSON(http.StatusOK, details)
	})

	r.POST("/api/chat", func(c *gin.Context) {
		// Read the raw request body
		rawBody, err := io.ReadAll(c.Request.Body)
		if err != nil {
			slog.Error("Failed to read raw request body", "Error", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body"})
			return
		}
		
		// Log the raw request body
		// slog.Info("Raw Chat request received", "raw_body", string(rawBody))
		
		// Define a custom struct specifically for message with images
		type MessageWithImages struct {
			Role    string   `json:"role"`
			Content string   `json:"content"`
			Images  []string `json:"images,omitempty"`
		}
		
		// Define a custom request struct that can properly handle images in messages
		type CustomChatRequest struct {
			Model    string             `json:"model"`
			Messages []MessageWithImages `json:"messages"`
			Stream   *bool              `json:"stream"`
			Images   []string           `json:"images,omitempty"`
			Options  map[string]interface{} `json:"options,omitempty"`
			KeepAlive int                `json:"keep_alive,omitempty"`
		}
		
		// Parse the raw JSON directly to catch images in messages
		var customRequest CustomChatRequest
		if err := json.Unmarshal(rawBody, &customRequest); err != nil {
			slog.Error("Failed to parse raw request", "Error", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}
		
		// Create our regular request struct
		var request struct {
			Model    string                         `json:"model"`
			Messages []openai.ChatCompletionMessage `json:"messages"`
			Stream   *bool                          `json:"stream"`
			Images   []string                       `json:"images,omitempty"`
		}
		
		// Fill in the standard fields
		request.Model = customRequest.Model
		request.Stream = customRequest.Stream
		request.Images = customRequest.Images
		
		// Convert custom messages to standard messages
		for _, customMsg := range customRequest.Messages {
			stdMsg := openai.ChatCompletionMessage{
				Role:    customMsg.Role,
				Content: customMsg.Content,
			}
			request.Messages = append(request.Messages, stdMsg)
		}
		
		// Log the entire request message
		requestJson, _ := json.MarshalIndent(request, "", "  ")
		// slog.Info("Chat request received", 
		// 	"model", request.Model,
		// 	"messagesCount", len(request.Messages),
		// 	"requestJson", string(requestJson))
		
		// Process images in messages from our custom parser
		for i, customMsg := range customRequest.Messages {
			// Skip if not a user message or no images
			if customMsg.Role != openai.ChatMessageRoleUser || len(customMsg.Images) == 0 {
				continue
			}
			
			// slog.Info("Images found within message from custom parser", 
			// 	"messageIndex", i,
			// 	"imageCount", len(customMsg.Images))
			
			// Process the images for this message
			msg := &request.Messages[i]
			prompt := msg.Content
			var contentItems []openai.ChatMessagePart
			
			// Add text content
			contentItems = append(contentItems, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeText,
				Text: prompt,
			})
			
			// Add image contents
			for imgIdx, imgBase64 := range customMsg.Images {
				// Validate image data isn't empty
				if len(imgBase64) == 0 {
					slog.Error("Empty image data received, skipping", 
						"messageIndex", i,
						"imageIndex", imgIdx)
					continue
				}
				
				// Debug the image data
				imgSize := len(imgBase64)
				// slog.Info("Processing image", 
				// 	"messageIndex", i,
				// 	"imageIndex", imgIdx,
				// 	"imageSize", imgSize,
				// 	"imagePrefix", imgBase64[:min(20, imgSize)])
				
				formattedURL := formatImageForAPI(imgBase64)
				// slog.Info("Formatted image URL", 
				// 	"urlPrefix", formattedURL[:min(50, len(formattedURL))])
				
				contentItems = append(contentItems, openai.ChatMessagePart{
					Type: openai.ChatMessagePartTypeImageURL,
					ImageURL: &openai.ChatMessageImageURL{
						URL: formattedURL,
					},
				})
				// slog.Info("Added image from message to multimodal message", 
				// 	"messageIndex", i,
				// 	"imageIndex", imgIdx, 
				// 	"imageSize", imgSize/1024, "KB")
			}
			
			// Replace the user message with the multimodal content
			msg.Content = "" // Will be ignored in favor of MultiContent
			msg.MultiContent = contentItems
			// slog.Info("Successfully converted message to multimodal format", 
			// 	"messageIndex", i,
			// 	"totalContentParts", len(contentItems))
		}
		
		// Process images if present in the top-level request and add them to the last user message
		if len(request.Images) > 0 && len(request.Messages) > 0 {
			// slog.Info("Images received in top-level /api/chat request", 
			// 	"count", len(request.Images), 
			// 	"model", request.Model,
			// 	"firstImageLength", len(request.Images[0])/1024, "KB")
			
			// Find the last user message
			lastUserMsgIndex := -1
			for i := len(request.Messages) - 1; i >= 0; i-- {
				if request.Messages[i].Role == openai.ChatMessageRoleUser {
					lastUserMsgIndex = i
					break
				}
			}
			
			// If we found a user message, add image content to it
			if lastUserMsgIndex >= 0 {
				userMsg := &request.Messages[lastUserMsgIndex]
				prompt := userMsg.Content
				
				// Check if this message already has MultiContent from previous processing
				var contentItems []openai.ChatMessagePart
				if len(userMsg.MultiContent) > 0 {
					// Use existing MultiContent
					contentItems = userMsg.MultiContent
					// slog.Info("Message already has MultiContent, appending to it", 
					// 	"messageIndex", lastUserMsgIndex,
					// 	"existingContentParts", len(contentItems))
				} else {
					// Create new MultiContent with text
					contentItems = append(contentItems, openai.ChatMessagePart{
						Type: openai.ChatMessagePartTypeText,
						Text: prompt,
					})
					
					// slog.Info("Adding top-level images to last user message", 
					// 	"messageIndex", lastUserMsgIndex, 
					// 	"originalContent", prompt[:min(50, len(prompt))])
				}
				
				// Add image contents
				for i, imgBase64 := range request.Images {
					contentItems = append(contentItems, openai.ChatMessagePart{
						Type: openai.ChatMessagePartTypeImageURL,
						ImageURL: &openai.ChatMessageImageURL{
							URL: formatImageForAPI(imgBase64),
						},
					})
					// slog.Info("Added top-level image to multimodal message", 
					// 	"imageIndex", i, 
					// 	"imageSize", len(imgBase64)/1024, "KB")
				}
				
				// Replace the user message with the multimodal content
				userMsg.Content = "" // Will be ignored in favor of MultiContent
				userMsg.MultiContent = contentItems
				// slog.Info("Successfully converted to multimodal message with top-level images", 
				// 	"totalContentParts", len(contentItems))
			}
		}

		// Определяем, нужен ли стриминг (по умолчанию true, если не указано для /api/chat)
		// ВАЖНО: Open WebUI может НЕ передавать "stream": true для /api/chat, подразумевая это.
		// Нужно проверить, какой запрос шлет Open WebUI. Если не шлет, ставим true.
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		// Если стриминг не запрошен, нужно будет реализовать отдельную логику
		// для сбора полного ответа и отправки его одним JSON.
		// Пока реализуем только стриминг.
		if !streamRequested {
			// Handle non-streaming response
			fullModelName, err := provider.GetFullModelName(request.Model)
			if err != nil {
				slog.Error("Error getting full model name", "Error", err)
				// Ollama returns 404 for invalid model names
				c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
				return
			}

			// Call Chat to get the complete response
			response, err := provider.Chat(request.Messages, fullModelName)
			if err != nil {
				slog.Error("Failed to get chat response", "Error", err)
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			// Format the response according to Ollama's format
			if len(response.Choices) == 0 {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "No response from model"})
				return
			}

			// Extract the content from the response
			content := ""
			if len(response.Choices) > 0 && response.Choices[0].Message.Content != "" {
				content = response.Choices[0].Message.Content
			}

			// Get finish reason, default to "stop" if not provided
			finishReason := "stop"
			if response.Choices[0].FinishReason != "" {
				finishReason = string(response.Choices[0].FinishReason)
			}

			// Create Ollama-compatible response
			ollamaResponse := map[string]interface{}{
				"model":             fullModelName,
				"created_at":        time.Now().Format(time.RFC3339),
				"message": map[string]string{
					"role":    "assistant",
					"content": content,
				},
				"done":              true,
				"finish_reason":     finishReason,
				"total_duration":    response.Usage.TotalTokens * 10, // Approximate duration based on token count
				"load_duration":     0,
				"prompt_eval_count": response.Usage.PromptTokens,
				"eval_count":        response.Usage.CompletionTokens,
				"eval_duration":     response.Usage.CompletionTokens * 10, // Approximate duration based on token count
			}

			c.JSON(http.StatusOK, ollamaResponse)
			return
		}

		slog.Info("Requested model", "model", request.Model)
		fullModelName, err := provider.GetFullModelName(request.Model)
		if err != nil {
			slog.Error("Error getting full model name", "Error", err, "model", request.Model)
			// Ollama возвращает 404 на неправильное имя модели
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}
		slog.Info("Using model", "fullModelName", fullModelName)

		// Call ChatStream to get the stream
		stream, err := provider.ChatStream(request.Messages, fullModelName)
		if err != nil {
			slog.Error("Failed to create stream", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer stream.Close() // Ensure stream closure

		// --- ИСПРАВЛЕНИЯ для NDJSON (Ollama-style) ---

		// Set headers CORRECTLY for Newline Delimited JSON
		c.Writer.Header().Set("Content-Type", "application/x-ndjson") // <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")
		// Transfer-Encoding: chunked устанавливается Gin автоматически

		w := c.Writer // Получаем ResponseWriter
		flusher, ok := w.(http.Flusher)
		if !ok {
			slog.Error("Expected http.ResponseWriter to be an http.Flusher")
			// Отправить ошибку клиенту уже сложно, т.к. заголовки могли уйти
			return
		}

		var lastFinishReason string

		// Stream responses back to the client
		for {
			response, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				// End of stream from the backend provider
				break
			}
			if err != nil {
				slog.Error("Backend stream error", "Error", err)
				// Попытка отправить ошибку в формате NDJSON
				// Ollama обычно просто обрывает соединение или шлет 500 перед этим
				errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
				errorJson, _ := json.Marshal(errorMsg)
				fmt.Fprintf(w, "%s\n", string(errorJson)) // Отправляем ошибку + \n
				flusher.Flush()
				return
			}

			// Сохраняем причину остановки, если она есть в чанке
			if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
				lastFinishReason = string(response.Choices[0].FinishReason)
			}

			// Build JSON response structure for intermediate chunks (Ollama chat format)
			responseJSON := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"message": map[string]string{
					"role":    "assistant",
					"content": response.Choices[0].Delta.Content, // Может быть ""
				},
				"done": false, // Всегда false для промежуточных чанков
			}

			// Marshal JSON
			jsonData, err := json.Marshal(responseJSON)
			if err != nil {
				slog.Error("Error marshaling intermediate response JSON", "Error", err)
				return // Прерываем, так как не можем отправить данные
			}

			// Send JSON object followed by a newline
			fmt.Fprintf(w, "%s\n", string(jsonData)) // <--- ИЗМЕНЕНО: Формат NDJSON (JSON + \n)

			// Flush data to send it immediately
			flusher.Flush()
		}

		// --- Отправка финального сообщения (done: true) в стиле Ollama ---

		// Определяем причину остановки (если бэкенд не дал, ставим 'stop')
		// Ollama использует 'stop', 'length', 'content_filter', 'tool_calls'
		if lastFinishReason == "" {
			lastFinishReason = "stop"
		}

		// ВАЖНО: Замените nil на 0 для числовых полей статистики
		finalResponse := map[string]interface{}{
			"model":             fullModelName,
			"created_at":        time.Now().Format(time.RFC3339),
			"message": map[string]string{
				"role":    "assistant",
				"content": "", // Пустой контент для финального сообщения
			},
			"done":              true,
			"finish_reason":     lastFinishReason, // Необязательно для /api/chat Ollama, но не вредит
			"total_duration":    0,
			"load_duration":     0,
			"prompt_eval_count": 0, // <--- ИЗМЕНЕНО: nil заменен на 0
			"eval_count":        0, // <--- ИЗМЕНЕНО: nil заменен на 0
			"eval_duration":     0,
		}

		finalJsonData, err := json.Marshal(finalResponse)
		if err != nil {
			slog.Error("Error marshaling final response JSON", "Error", err)
			return
		}

		// Отправляем финальный JSON-объект + newline
		fmt.Fprintf(w, "%s\n", string(finalJsonData)) // <--- ИЗМЕНЕНО: Формат NDJSON
		flusher.Flush()

		// ВАЖНО: Для NDJSON НЕТ 'data: [DONE]' маркера.
		// Клиент понимает конец потока по получению объекта с "done": true
		// и/или по закрытию соединения сервером (что Gin сделает автоматически после выхода из хендлера).

		// --- Конец исправлений ---
	})

	// --- Chat API endpoint (existing code) ---

	r.POST("/api/generate", func(c *gin.Context) {
		// Read the raw request body
		rawBody, err := io.ReadAll(c.Request.Body)
		if err != nil {
			slog.Error("Failed to read raw request body", "Error", err)
			c.JSON(http.StatusBadRequest, gin.H{"error": "Failed to read request body"})
			return
		}
		
		// Log the raw request body
		// slog.Info("Raw Generate request received", "raw_body", string(rawBody))
		
		// Restore the request body for later binding
		c.Request.Body = io.NopCloser(bytes.NewBuffer(rawBody))
		
		var request struct {
			Model    string   `json:"model"`
			Prompt   string   `json:"prompt"`
			System   string   `json:"system,omitempty"`
			Stream   *bool    `json:"stream"`
			Raw      bool     `json:"raw,omitempty"`
			Images   []string `json:"images,omitempty"`
			Format   string   `json:"format,omitempty"`
			Options  map[string]interface{} `json:"options,omitempty"`
			Template string   `json:"template,omitempty"`
			Context  []int    `json:"context,omitempty"`
			KeepAlive string  `json:"keep_alive,omitempty"`
		}

		// Parse the JSON request
		if err := c.ShouldBindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid JSON payload"})
			return
		}
		
		// Log the entire request message
		requestJson, _ := json.MarshalIndent(request, "", "  ")
		// slog.Info("Generate request received", 
		// 	"model", request.Model,
		// 	"promptLength", len(request.Prompt),
		// 	"hasImages", len(request.Images) > 0,
		// 	"requestJson", string(requestJson))
		
		// Log image information if present
		// if len(request.Images) > 0 {
		// 	slog.Info("Images received in /api/generate request", 
		// 		"count", len(request.Images), 
		// 		"model", request.Model,
		// 		"firstImageLength", len(request.Images[0])/1024, "KB",
		// 		"promptLength", len(request.Prompt))
			
		// 	for i, img := range request.Images {
		// 		slog.Info("Image details", 
		// 			"imageIndex", i, 
		// 			"imageSize", len(img)/1024, "KB")
		// 	}
		// }

		// Determine if streaming is requested (default to true if not specified)
		streamRequested := true
		if request.Stream != nil {
			streamRequested = *request.Stream
		}

		// Get the full model name from the provider
		slog.Info("Requested model", "model", request.Model)
		fullModelName, err := provider.GetFullModelName(request.Model)
		if err != nil {
			slog.Error("Error getting full model name", "Error", err, "model", request.Model)
			c.JSON(http.StatusNotFound, gin.H{"error": err.Error()})
			return
		}
		slog.Info("Using model", "fullModelName", fullModelName)

		// Handle non-streaming request
		if !streamRequested {
			// Call Generate to get a complete response
			response, err := provider.Generate(request.Prompt, fullModelName, request.System, request.Images)
			if err != nil {
				slog.Error("Failed to get generate response", "Error", err)
				c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
				return
			}

			// Format the response according to Ollama's format
			if len(response.Choices) == 0 {
				c.JSON(http.StatusInternalServerError, gin.H{"error": "No response from model"})
				return
			}

			// Extract content from the response
			content := ""
			if len(response.Choices) > 0 && response.Choices[0].Message.Content != "" {
				content = response.Choices[0].Message.Content
			}

			// Get finish reason, default to "stop" if not provided
			finishReason := "stop"
			if response.Choices[0].FinishReason != "" {
				finishReason = string(response.Choices[0].FinishReason)
			}

			// Create Ollama-compatible response
			ollamaResponse := map[string]interface{}{
				"model":               fullModelName,
				"created_at":          time.Now().Format(time.RFC3339),
				"response":            content,
				"done":                true,
				"done_reason":         finishReason,
				"context":             []int{1, 2, 3}, // Placeholder context
				"total_duration":      response.Usage.TotalTokens * 10000000, // Approximate duration in ns
				"load_duration":       5000000, // Placeholder 5ms in ns
				"prompt_eval_count":   response.Usage.PromptTokens,
				"prompt_eval_duration": response.Usage.PromptTokens * 10000000, // Approximate
				"eval_count":          response.Usage.CompletionTokens,
				"eval_duration":       response.Usage.CompletionTokens * 10000000, // Approximate
			}

			c.JSON(http.StatusOK, ollamaResponse)
			return
		}

		// Handle streaming request
		stream, err := provider.GenerateStream(request.Prompt, fullModelName, request.System, request.Images)
		if err != nil {
			slog.Error("Failed to create generate stream", "Error", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
			return
		}
		defer stream.Close()

		// Set headers for NDJSON streaming response
		c.Writer.Header().Set("Content-Type", "application/x-ndjson")
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")

		w := c.Writer
		flusher, ok := w.(http.Flusher)
		if !ok {
			slog.Error("Expected http.ResponseWriter to be an http.Flusher")
			return
		}

		var lastFinishReason string

		// Stream responses back to the client in Ollama's format
		for {
			response, err := stream.Recv()
			if errors.Is(err, io.EOF) {
				// End of stream
				break
			}
			if err != nil {
				slog.Error("Backend stream error", "Error", err)
				errorMsg := map[string]string{"error": "Stream error: " + err.Error()}
				errorJson, _ := json.Marshal(errorMsg)
				fmt.Fprintf(w, "%s\n", string(errorJson))
				flusher.Flush()
				return
			}

			// Save finish reason if available
			if len(response.Choices) > 0 && response.Choices[0].FinishReason != "" {
				lastFinishReason = string(response.Choices[0].FinishReason)
			}

			// Build JSON response structure for intermediate chunks (Ollama generate format)
			responseJSON := map[string]interface{}{
				"model":      fullModelName,
				"created_at": time.Now().Format(time.RFC3339),
				"response":   response.Choices[0].Delta.Content,
				"done":       false,
			}

			// Marshal and send
			jsonData, err := json.Marshal(responseJSON)
			if err != nil {
				slog.Error("Error marshaling intermediate response JSON", "Error", err)
				return
			}

			fmt.Fprintf(w, "%s\n", string(jsonData))
			flusher.Flush()
		}

		// Send final message with done=true and stats
		if lastFinishReason == "" {
			lastFinishReason = "stop"
		}

		finalResponse := map[string]interface{}{
			"model":               fullModelName,
			"created_at":          time.Now().Format(time.RFC3339),
			"response":            "",
			"done":                true,
			"done_reason":         lastFinishReason,
			"context":             []int{1, 2, 3}, // Placeholder context
			"total_duration":      1000000000, // Placeholder 1s in ns
			"load_duration":       5000000, // Placeholder 5ms in ns
			"prompt_eval_count":   20, // Placeholder
			"prompt_eval_duration": 200000000, // Placeholder 200ms in ns
			"eval_count":          100, // Placeholder
			"eval_duration":       800000000, // Placeholder 800ms in ns
		}

		finalJsonData, err := json.Marshal(finalResponse)
		if err != nil {
			slog.Error("Error marshaling final response JSON", "Error", err)
			return
		}

		fmt.Fprintf(w, "%s\n", string(finalJsonData))
		flusher.Flush()
	})

	r.Run(":11434")
}
