#pragma once
#include <vector>
#include <unordered_map>
#include "cbow.hpp"

template <typename ModelType>
class Embedding : public torch::nn::Module {
public:
    Embedding(int64_t vocab_size, short window_size, int32_t embedding_dim, float lr, std::string device);
    ~Embedding() = default;
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor operator()(torch::Tensor input);
    torch::Tensor operator[](std::u32string word);
    void fit(torch::Tensor data, short batch_size, int64_t num_epochs=10, size_t num_workers=1);
    std::vector<std::u32string> tokenize(std::u32string &text);
    void make_vocab(std::vector<std::u32string> &tokens);
    torch::Tensor text_to_idx(std::u32string &text);
    std::unordered_map<std::u32string, int> vocab;
private:
    std::unique_ptr<Model> model;
};