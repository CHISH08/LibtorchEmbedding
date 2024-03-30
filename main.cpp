#include "embedding.hpp"
#include <unistd.h>
#include <codecvt>
#include <sstream>
#include <locale>

std::u32string readFileToU32String(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Не удалось открыть файл");
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    std::u32string u32content = converter.from_bytes(content);

    return u32content;
}

int main() {
    short num_cores = sysconf(_SC_NPROCESSORS_CONF);
    short embedding_dim = 150, window_size = 10, batch_size = 2650, num_epochs = 20;
    double lr = 5e-3;
    std::locale::global(std::locale(""));
    torch::manual_seed(123);

    if (torch::cuda::is_available()) {
        torch::cuda::manual_seed(123);
    }

    std::string device = "cuda";
    std::u32string text = readFileToU32String("../voina_i_mir.txt");
    for (int ws = 1; ws <= 100; ws += 10) {
        for (int e_dim = 10; e_dim <= 310; e_dim += 60) {
            Embedding<CBOW> model(text, ws, e_dim, lr, device);
            torch::Tensor idxs = model.text_to_idx(text);
            model.fit(idxs, batch_size, num_epochs, num_cores);
        }
    }
    // std::u32string test_word = U"андрей";
    // std::vector<std::pair<std::u32string, torch::Tensor>> ans = model.k_nearest(test_word, 10, 1, 0);
    // std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> convert;
    // for (auto item: ans) {
    //     std::string test_word_utf8 = convert.to_bytes(item.first);
    //     std::cout << test_word_utf8 << ":\n" << item.second << "\n";
    // }
    // ans = model.k_nearest(test_word, 10, 0, 0);
    // for (auto item: ans) {
    //     std::string test_word_utf8 = convert.to_bytes(item.first);
    //     std::cout << test_word_utf8 << ":\n" << item.second << "\n";
    // }
    return 0;
}
