#pragma once
#include <fstream>
#include <string>

namespace nndemo {
class FileReader {
public:
    FileReader(const char* filename);
    ~FileReader();
    // returns strbuf filled with next line in file
    std::string& getline(std::string& strbuf);
    bool good() const;
    bool eof() const;
private:
    std::ifstream ifh;
    // no copying
    FileReader(const FileReader& other) {}
};
}
