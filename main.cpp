#include "embedding.hpp"
#include <unistd.h>
#include <codecvt>
#include <locale>

int main() {
    short num_cores = sysconf(_SC_NPROCESSORS_CONF);
    short embedding_dim = 20, window_size = 1, batch_size = 100, num_epochs = 10;
    double lr = 0.01;
    std::locale::global(std::locale(""));

    std::string device = "cpu";
    std::u32string text = U"Мама мыла раму, а папа ломал раму";
    Embedding<CBOW> model(text, window_size, embedding_dim, lr, device);
    torch::Tensor idxs = model.text_to_idx(text);
    model.fit(idxs, batch_size, num_epochs);
    std::u32string test_word = U"мама";
    std::vector<std::pair<std::u32string, torch::Tensor>> ans = model.k_nearest(test_word, 2);
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    for (auto item: ans) {
        std::string test_word_utf8 = convert.to_bytes(item.first);
        std::cout << test_word_utf8 << ":\n" << item.second << "\n";
    }
    return 0;
}
