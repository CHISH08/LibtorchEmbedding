#include "embedding.hpp"

char32_t tolower_utf32(char32_t ch) {
    if (ch >= U'A' && ch <= U'Z') {
        return ch - U'A' + U'a';
    }
    if (ch >= U'А' && ch <= U'Я') {
        return ch - U'А' + U'а';
    }
    if (ch == U'Ё') {
        return U'ё';
    }
    return ch;
}

template <typename ModelType>
Embedding<ModelType>::Embedding(std::u32string &text, short window_size, int32_t embedding_dim, float lr, std::string device) {
    std::vector<std::u32string> tokens = this->tokenize(text);
    this->make_vocab(tokens);
    this->model = std::make_unique<ModelType>(this->vocab.size(), window_size, embedding_dim, lr, device);
}

template <typename ModelType>
torch::Tensor Embedding<ModelType>::forward(torch::Tensor x) {
    return this->model->forward(x);
}

template <typename ModelType>
torch::Tensor Embedding<ModelType>::operator()(torch::Tensor input) {
    return this->model->operator()(input);
}

template <typename ModelType>
torch::Tensor Embedding<ModelType>::operator[](std::u32string word) {
    torch::Tensor word_idx = torch::tensor(this->vocab[word]);
    return this->model->operator()(word_idx);
}

template <typename ModelType>
void Embedding<ModelType>::fit(torch::Tensor data, short batch_size, int64_t num_epochs, size_t num_workers) {
    this->model->fit(data, batch_size, num_epochs, num_workers);
}

template <typename ModelType>
std::vector<std::u32string> Embedding<ModelType>::tokenize(std::u32string &text) {
    text += U'\n';
    std::vector<std::u32string> tokens;
    std::u32string cur_str = U"";
    for (int64_t i = 0; i < text.size(); ++i) {
        if ((text[i] >= U'а' && text[i] <= U'я') || (text[i] >= U'А' && text[i] <= U'Я') || (text[i] >= U'a' && text[i] <= U'z') || (text[i] >= U'A' && text[i] <= U'Z') || (text[i] >= U'0' && text[i] <= U'9') || ((text[i] == U'\'') && (cur_str.size() != 0)) || (text[i] == U'ё') || (text[i] == U'Ё')) {
            cur_str += tolower_utf32(text[i]);
        } else if (cur_str.size() != 0) {
            tokens.push_back(cur_str);
            cur_str = U"";
        }
    }
    return tokens;
}

template <typename ModelType>
void Embedding<ModelType>::make_vocab(std::vector<std::u32string> &tokens) {
    int64_t i = this->vocab.size();
    for (auto str: tokens) {
        if (this->vocab.find(str) == this->vocab.end()) {
            this->vocab[str] = i;
            ++i;
        }
    }
}

template <typename ModelType>
torch::Tensor Embedding<ModelType>::text_to_idx(std::u32string &text) {
    std::vector<std::u32string> tokens = this->tokenize(text);
    torch::Tensor token = torch::zeros({static_cast<long>(tokens.size())}, torch::dtype(torch::kInt32));
    int64_t i = 0;
    for (auto str: tokens) {
        token[i] = this->vocab[tokens[i]];
        ++i;
    }
    return token;
}

template <typename ModelType>
std::vector<std::pair<std::u32string, torch::Tensor>> Embedding<ModelType>::k_nearest(std::u32string &word, int k) {
    std::vector<std::pair<std::u32string, torch::Tensor>> k_nearest_word;
    return k_nearest_word;
}


template class Embedding<CBOW>;