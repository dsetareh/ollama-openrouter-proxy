package main

import (
	"context"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
)

// Helper function to ensure image has proper format for OpenRouter API
func formatImageForAPI(imgBase64 string) string {
	// If the image string is empty, return empty
	if len(imgBase64) == 0 {
		slog.Error("Empty image data received in formatImageForAPI")
		return ""
	}
	
	// Trim any whitespace that might be present
	imgBase64 = strings.TrimSpace(imgBase64)
	
	// Check if image already has a data URL prefix
	if strings.HasPrefix(imgBase64, "data:image/") && strings.Contains(imgBase64, ";base64,") {
		// Already has proper format
		slog.Info("Image already has proper data URL format")
		return imgBase64
	}
	
	// Log the start of the image for debugging
	slog.Info("Image format detection", 
		"imgStart", imgBase64[:min(30, len(imgBase64))])
	
	// If it starts with '/', it might be using the standard base64 encoding marker for JPEG
	if strings.HasPrefix(imgBase64, "/9j/") {
		slog.Info("Detected JPEG image format from /9j/ prefix")
		return "data:image/jpeg;base64," + imgBase64
	} else if strings.HasPrefix(imgBase64, "iVBOR") {
		// This is likely a PNG image (PNG header)
		slog.Info("Detected PNG image format")
		return "data:image/png;base64," + imgBase64
	} else if strings.HasPrefix(imgBase64, "R0lGOD") {
		// This is likely a GIF image
		slog.Info("Detected GIF image format")
		return "data:image/gif;base64," + imgBase64
	} else if strings.HasPrefix(imgBase64, "UklGR") {
		// This is likely a WEBP image
		slog.Info("Detected WEBP image format")
		return "data:image/webp;base64," + imgBase64
	} else {
		// If we can't determine the type, default to JPEG
		slog.Info("Could not determine image type, defaulting to JPEG", 
			"imageStart", imgBase64[:min(20, len(imgBase64))])
		return "data:image/jpeg;base64," + imgBase64
	}
}

type OpenrouterProvider struct {
	client     *openai.Client
	modelNames []string // Shared storage for model names
}

func NewOpenrouterProvider(apiKey string) *OpenrouterProvider {
	config := openai.DefaultConfig(apiKey)
	// config.BaseURL = "https://openrouter.ai/api/v1/" // Custom endpoint if needed

	// Get BaseURL from environment variable
	baseURL := os.Getenv("OPENROUTER_BASE_URL")
	if baseURL != "" {
		config.BaseURL = baseURL
		slog.Info("Using custom BaseURL from environment variable", "baseURL", baseURL)
	} else {
		config.BaseURL = "https://openrouter.ai/api/v1/" // Default endpoint
		slog.Info("Using default BaseURL", "baseURL", config.BaseURL)
	}
	
	// Get header values from environment variables
	httpReferer := os.Getenv("OPENROUTER_HTTP_REFERER")
	if httpReferer == "" {
		httpReferer = "" // Default value if env var not set
		slog.Info("OPENROUTER_HTTP_REFERER not set, using default value")
	}
	
	xTitle := os.Getenv("OPENROUTER_X_TITLE") 
	if xTitle == "" {
		xTitle = "ollama-proxy" // Default value if env var not set
		slog.Info("OPENROUTER_X_TITLE not set, using default value")
	}
	
	// Add custom headers for OpenRouter
	config.HTTPClient = &http.Client{
		Transport: &headerTransport{
			base: http.DefaultTransport,
			headers: map[string]string{
				"HTTP-Referer": httpReferer,
				"X-Title":      xTitle,
			},
		},
	}
	
	return &OpenrouterProvider{
		client:     openai.NewClientWithConfig(config),
		modelNames: []string{},
	}
}

// Custom transport to add headers to all requests
type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t *headerTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	for key, value := range t.headers {
		req.Header.Add(key, value)
	}
	return t.base.RoundTrip(req)
}

func (o *OpenrouterProvider) Chat(messages []openai.ChatCompletionMessage, modelName string) (openai.ChatCompletionResponse, error) {
	// Create a chat completion request
	req := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   false,
	}

	// Call the OpenAI API to get a complete response
	resp, err := o.client.CreateChatCompletion(context.Background(), req)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	// Return the complete response
	return resp, nil
}

func (o *OpenrouterProvider) ChatStream(messages []openai.ChatCompletionMessage, modelName string) (*openai.ChatCompletionStream, error) {
	// Log the messages being sent for debugging
	slog.Info("Sending messages to OpenRouter", "messageCount", len(messages))
	for i, msg := range messages {
		if msg.MultiContent != nil && len(msg.MultiContent) > 0 {
			slog.Info("Message with MultiContent", 
				"index", i, 
				"role", msg.Role, 
				"contentPartCount", len(msg.MultiContent))
			
			for j, part := range msg.MultiContent {
				if part.Type == openai.ChatMessagePartTypeImageURL && part.ImageURL != nil {
					urlPreview := ""
					if len(part.ImageURL.URL) > 50 {
						urlPreview = part.ImageURL.URL[:50] + "..."
					} else {
						urlPreview = part.ImageURL.URL
					}
					slog.Info("Image content part", 
						"messageIndex", i,
						"partIndex", j, 
						"urlPreview", urlPreview)
				}
			}
		}
	}

	// Create a chat completion request
	req := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   true,
	}

	// Call the OpenAI API to get a streaming response
	stream, err := o.client.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		return nil, err
	}

	// Return the stream for further processing
	return stream, nil
}

// Generate creates a completion (non-streaming) for a text prompt
func (o *OpenrouterProvider) Generate(prompt string, modelName string, systemPrompt string, images []string) (openai.ChatCompletionResponse, error) {
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		},
	}

	// Add system message if provided
	if systemPrompt != "" {
		messages = append([]openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
		}, messages...)
	}

	// Add images if provided (for multimodal models)
	if len(images) > 0 {
		var contentItems []openai.ChatMessagePart

		// Add text content
		contentItems = append(contentItems, openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: prompt,
		})

		// Add image contents with proper formatting
		for idx, imgBase64 := range images {
			// Ensure image isn't empty
			if len(imgBase64) == 0 {
				slog.Error("Empty image data received, skipping", "imageIndex", idx)
				continue
			}
			
			// Format the image URL correctly
			formattedURL := formatImageForAPI(imgBase64)
			
			// Log what we're sending
			slog.Info("Adding image to Generate request", 
				"imageIndex", idx, 
				"formattedUrlPrefix", formattedURL[:min(50, len(formattedURL))])
			
			contentItems = append(contentItems, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL: formattedURL,
				},
			})
		}

		// Replace the user message with the one containing content parts
		messages[len(messages)-1] = openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: "", // Will be ignored in favor of ContentParts
			MultiContent: contentItems,
		}
		
		slog.Info("Successfully prepared multimodal message for Generate", 
			"contentPartCount", len(contentItems))
	}

	// Create a chat completion request
	req := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   false,
	}

	// Call the OpenAI API to get a complete response
	resp, err := o.client.CreateChatCompletion(context.Background(), req)
	if err != nil {
		return openai.ChatCompletionResponse{}, err
	}

	// Return the complete response
	return resp, nil
}

// GenerateStream creates a streaming completion for a text prompt
func (o *OpenrouterProvider) GenerateStream(prompt string, modelName string, systemPrompt string, images []string) (*openai.ChatCompletionStream, error) {
	messages := []openai.ChatCompletionMessage{
		{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		},
	}

	// Add system message if provided
	if systemPrompt != "" {
		messages = append([]openai.ChatCompletionMessage{
			{
				Role:    openai.ChatMessageRoleSystem,
				Content: systemPrompt,
			},
		}, messages...)
	}

	// Add images if provided (for multimodal models)
	if len(images) > 0 {
		var contentItems []openai.ChatMessagePart

		// Add text content
		contentItems = append(contentItems, openai.ChatMessagePart{
			Type: openai.ChatMessagePartTypeText,
			Text: prompt,
		})

		// Add image contents with proper formatting
		for idx, imgBase64 := range images {
			// Ensure image isn't empty
			if len(imgBase64) == 0 {
				slog.Error("Empty image data received, skipping", "imageIndex", idx)
				continue
			}
			
			// Format the image URL correctly
			formattedURL := formatImageForAPI(imgBase64)
			
			// Log what we're sending
			slog.Info("Adding image to GenerateStream request", 
				"imageIndex", idx, 
				"formattedUrlPrefix", formattedURL[:min(50, len(formattedURL))])
			
			contentItems = append(contentItems, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL: formattedURL,
				},
			})
		}

		// Replace the user message with the one containing content parts
		messages[len(messages)-1] = openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleUser,
			Content: "", // Will be ignored in favor of ContentParts
			MultiContent: contentItems,
		}
		
		slog.Info("Successfully prepared multimodal message for GenerateStream", 
			"contentPartCount", len(contentItems))
	}

	// Create a chat completion request
	req := openai.ChatCompletionRequest{
		Model:    modelName,
		Messages: messages,
		Stream:   true,
	}

	// Call the OpenAI API to get a streaming response
	stream, err := o.client.CreateChatCompletionStream(context.Background(), req)
	if err != nil {
		return nil, err
	}

	// Return the stream for further processing
	return stream, nil
}

type ModelDetails struct {
	ParentModel       string   `json:"parent_model"`
	Format            string   `json:"format"`
	Family            string   `json:"family"`
	Families          []string `json:"families"`
	ParameterSize     string   `json:"parameter_size"`
	QuantizationLevel string   `json:"quantization_level"`
}

type Model struct {
	Name       string       `json:"name"`
	Model      string       `json:"model,omitempty"`
	ModifiedAt string       `json:"modified_at,omitempty"`
	Size       int64        `json:"size,omitempty"`
	Digest     string       `json:"digest,omitempty"`
	Details    ModelDetails `json:"details,omitempty"`
}

func (o *OpenrouterProvider) GetModels() ([]Model, error) {
	currentTime := time.Now().Format(time.RFC3339)

	// Fetch models from the OpenAI API
	modelsResponse, err := o.client.ListModels(context.Background())
	if err != nil {
		return nil, err
	}

	// Clear shared model storage
	o.modelNames = []string{}

	var models []Model
	for _, apiModel := range modelsResponse.Models {
		// Split model name
		parts := strings.Split(apiModel.ID, "/")
		name := parts[len(parts)-1]

		// Store name in shared storage
		o.modelNames = append(o.modelNames, apiModel.ID)

		// Create model struct
		model := Model{
			Name:       name,
			Model:      name,
			ModifiedAt: currentTime,
			Size:       0, // Stubbed size
			Digest:     name,
			Details: ModelDetails{
				ParentModel:       "",
				Format:            "gguf",
				Family:            "claude",
				Families:          []string{"claude"},
				ParameterSize:     "175B",
				QuantizationLevel: "Q4_K_M",
			},
		}
		models = append(models, model)
	}

	return models, nil
}

func (o *OpenrouterProvider) GetModelDetails(modelName string) (map[string]interface{}, error) {
	// Stub response; replace with actual model details if available
	currentTime := time.Now().Format(time.RFC3339)
	return map[string]interface{}{
		"license":    "STUB License",
		"system":     "STUB SYSTEM",
		"modifiedAt": currentTime,
		"details": map[string]interface{}{
			"format":             "gguf",
			"parameter_size":     "200B",
			"quantization_level": "Q4_K_M",
		},
		"model_info": map[string]interface{}{
			"architecture":    "STUB",
			"context_length":  200000,
			"parameter_count": 200_000_000_000,
		},
	}, nil
}

func (o *OpenrouterProvider) GetFullModelName(alias string) (string, error) {
	// If modelNames is empty or not populated yet, try to get models first
	if len(o.modelNames) == 0 {
		_, err := o.GetModels()
		if err != nil {
			return "", fmt.Errorf("failed to get models: %w", err)
		}
	}

	// First try exact match
	for _, fullName := range o.modelNames {
		if fullName == alias {
			return fullName, nil
		}
	}

	// Then try suffix match
	for _, fullName := range o.modelNames {
		if strings.HasSuffix(fullName, alias) {
			return fullName, nil
		}
	}

	// If no match found, just use the alias as is
	// This allows direct use of model names that might not be in the list
	return alias, nil
}

// Helper function for min (for Go versions that don't have it built-in)
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
