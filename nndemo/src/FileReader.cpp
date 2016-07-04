#include "nndemo/FileReader.h"

nndemo::FileReader::FileReader(const char* filename)
    :
    ifh(std::ifstream(filename, std::ifstream::in)) {}

nndemo::FileReader::~FileReader() {
    ifh.close();
}

std::string& nndemo::FileReader::getline(std::string& strbuf) {
    std::getline(ifh, strbuf);
    return strbuf;
}

bool nndemo::FileReader::good() const {
    return ifh.good();
}

bool nndemo::FileReader::eof() const {
    return ifh.eof();
}
