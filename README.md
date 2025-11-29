# AI-project-
#include <iostream>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <cmath> // For TF-IDF calculations (if implementing custom vectorizer)

// Function to clean and tokenize text
std::vector<std::string> preprocess_text(const std::string& raw_text) {
    std::string text = raw_text;
    std::vector<std::string> tokens;

    // 1. Text Cleaning: Convert to lowercase and remove punctuation
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    text.erase(std::remove_if(text.begin(), text.end(), ::ispunct), text.end());

    // 2. Tokenization: Split by whitespace
    std::stringstream ss(text);
    std::string word;
    while (ss >> word) {
        // 3. Stop Word Removal (requires a pre-loaded std::unordered_set<std::string> stop_words)
        // 4. Stemming/Lemmatization (highly complex in C++, usually omitted or done via a library)
        tokens.push_back(word);
    }
    return tokens;
}
/ This class simulates the Scikit-learn TfidfVectorizer logic
class TfidfVectorizerCpp {
private:
    std::map<std::string, int> vocabulary_;       // Word -> Index map (Loaded from Python)
    std::vector<double> idf_values_;             // IDF values (Loaded from Python)
    int feature_count_;

public:
    // Constructor would load vocabulary and IDF values from files (e.g., CSV, JSON)
    TfidfVectorizerCpp(const std::string& vocab_file, const std::string& idf_file);

    // Converts cleaned tokens into a feature vector
    std::vector<double> transform(const std::vector<std::string>& tokens) {
        std::vector<double> feature_vector(feature_count_, 0.0);
        std::map<std::string, int> term_frequency;

        // Calculate Term Frequency (TF)
        for (const std::string& token : tokens) {
            term_frequency[token]++;
        }

        // Calculate TF-IDF
        int total_words = tokens.size();
        for (const auto& pair : term_frequency) {
            const std::string& word = pair.first;
            int count = pair.second;

            if (vocabulary_.count(word)) {
                int index = vocabulary_[word];
                double tf = (double)count / total_words;
                double idf = idf_values_[index];
                feature_vector[index] = tf * idf; // TF * IDF
            }
        }
        return feature_vector;
    }
};
#include "TextPreprocessor.h"
#include "Vectorizer.h"
// #include <onnxruntime_cxx_api.h> // Include ONNX library

// ... (Instantiate the preprocessor and vectorizer)
// TextPreprocessor preprocessor;
// TfidfVectorizerCpp vectorizer("vocab.txt", "idf.txt");

int main() {
    std::string new_review = "This is a truly awful product.";

    // 1. Preprocess
    std::vector<std::string> tokens = preprocess_text(new_review);

    // 2. Vectorize
    std::vector<double> input_vector = vectorizer.transform(tokens);
    // input_vector is now the numerical feature set for the model

    // 3. Load and Predict with ONNX Runtime
    // Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestEnvironment");
    // Ort::Session session(env, "model.onnx", Ort::SessionOptions{nullptr});

    // Create tensor from input_vector...
    // Run session...
    // Get output tensor...

    // SIMULATION: Assume prediction is 0 (Negative)
    int prediction = 0;
    
    // 4. Display Result
    std::string result = (prediction == 2) ? "Positive" : (prediction == 1) ? "Neutral" : "Negative";
    std::cout << "Review: " << new_review << std::endl;
    std::cout << "Predicted Sentiment: " << result << std::endl;

    return 0;
}
