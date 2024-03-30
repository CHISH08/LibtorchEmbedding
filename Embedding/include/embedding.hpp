#pragma once
#include <vector>
#include <unordered_map>
#include <utility>
#include <fstream>
#include <filesystem>
#include "cbow.hpp"

template <typename ModelType>
class Embedding : public torch::nn::Module {
public:
    Embedding(std::u32string &text, short window_size, int32_t embedding_dim, float lr, std::string device);
    ~Embedding() = default;
    torch::Tensor forward(torch::Tensor x);
    torch::Tensor operator()(torch::Tensor input);
    torch::Tensor operator[](std::u32string &word);
    void fit(torch::Tensor &data, short batch_size, int64_t num_epochs=10, size_t num_workers=1, std::string file_path = "./train_metric");
    std::vector<std::u32string> tokenize(std::u32string &text);
    void make_vocab(std::vector<std::u32string> &tokens);
    torch::Tensor text_to_idx(std::u32string &text);
    std::vector<std::pair<std::u32string, torch::Tensor>> k_nearest(std::u32string &word, int k, bool cosin=0, bool out_emb=0);
    void to(std::string device);
    void metric_to_file(std::pair<std::vector<double>, std::vector<double>> &loss_met, int64_t window_size, int64_t embedding_dim, std::string file_path);
private:
    std::unordered_map<std::u32string, int> vocab;
    std::vector<std::u32string> vocab_word;
    int64_t vocab_size = 0;
    std::unique_ptr<Model> model;
    int32_t embedding_dim;
};
