// class.cpp
#include "lib.hpp"

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    torch::Tensor tokens;
    short window_size;
    int64_t vocab_size;

public:
    CustomDataset(torch::Tensor tokens, short window_size, int64_t vocab_size)
        : tokens(tokens), window_size(window_size), vocab_size(vocab_size) {}

    torch::data::Example<> get(size_t index) override {
        torch::Tensor left_context = tokens.slice(0, index, index + this->window_size);
        torch::Tensor right_context = tokens.slice(0, index + this->window_size + 1, index + 2 * this->window_size + 1);
        torch::Tensor context = torch::cat({left_context, right_context}, 0);
        torch::Tensor center = torch::zeros({vocab_size});
        center[tokens[index + this->window_size]] = 1;
        return {context, center};
    };

    torch::optional<size_t> size() const override {
        return tokens.size(0) - 2 * this->window_size;
    };
};

CBOW::CBOW(int64_t vocab_size, short window_size, int32_t embedding_dim, float lr, std::string device):
    embeddings(register_module("embeddings", torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, embedding_dim)))),
    linear(register_module("linear", torch::nn::Linear(torch::nn::LinearOptions(embedding_dim, vocab_size)))) {
    this->window_size = window_size;
    this->vocab_size = vocab_size;
    this->optim = std::make_unique<torch::optim::Adam>(this->parameters(), torch::optim::AdamOptions(lr));
    this->device = device;
    this->to(device);
}

torch::Tensor CBOW::forward(torch::Tensor x) {
    x = this->embeddings(x);
    x = torch::sum(x, /*axis=*/1);
    x = this->linear(x);
    x = torch::softmax(x, /*dim=*/1);
    return x;
}

torch::Tensor CBOW::operator()(torch::Tensor input) {
    return forward(input);
}

void CBOW::fit(torch::Tensor data, short batch_size, int64_t num_epochs, size_t num_workers){
    auto dataset = CustomDataset(data, window_size, this->vocab_size).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(
        dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers)
    );
    torch::nn::CrossEntropyLoss criterion;
    this->train();
    for (int64_t epoch = 1; epoch <= num_epochs; ++epoch) {
        double sum_loss = 0;
        for (auto& batch : *data_loader) {
            auto context = batch.data.to(this->device);
            auto center = batch.target.to(this->device);
            this->optim->zero_grad();

            auto outputs = this->forward(context);

            auto loss = criterion(outputs, center);
            loss.backward();
            this->optim->step();

            sum_loss += loss.item<double>();
        }
        std::cout << "Эпоха " << epoch << ", Loss: " << sum_loss << std::endl;
    }
    std::cout << "Обучение завершено." << std::endl;
}
