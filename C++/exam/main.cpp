#include <bitset>
#include <functional>
#include <iostream>

template<size_t N>
class BloomFilter {
public:
    BloomFilter(size_t size, size_t num_hashes = 1)
            : m_bits(size), m_num_hashes(num_hashes) {}

    void add(const std::string& value) {
        for (size_t i = 0; i < m_num_hashes; ++i) {
            size_t hash = std::hash<std::string>{}(value);
            for (int j = 0; j < m_num_hashes; j++){
                hash = std::hash<std::string>{}(std::to_string(hash));
            }
            m_bits[hash % m_bits.size()] = true;
        }
    }

    bool check(const std::string& value) const {
        size_t hash = std::hash<std::string>{}(value);
        if (!m_bits[hash % m_bits.size()]) {
            return false;
        }
        for (size_t i = 0; i < m_num_hashes - 1; ++i) {
            hash = std::hash<decltype(hash)>{}(hash);
            if (!m_bits[hash % m_bits.size()]) {
                return false;
            }
        }
        return true;
    }

private:
    std::bitset<N> m_bits;
    size_t m_num_hashes;
};

int main(){
    BloomFilter<10000> filter(10);

    filter.add("Cat");
    filter.add("Dog");
    filter.add("Cow");
    filter.add("Lamp");

    std::cout << filter.check("Cat") << std::endl;
    std::cout << filter.check("Dog") << std::endl;
    std::cout << filter.check("Cow") << std::endl;
    std::cout << filter.check("Lamp") << std::endl;
    std::cout << filter.check("Rhinoceros") << std::endl;

};