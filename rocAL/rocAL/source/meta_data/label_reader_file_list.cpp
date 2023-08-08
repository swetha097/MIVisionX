/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "commons.h"
#include "exception.h"
#include "label_reader_file_list.h"


using namespace std;

namespace filesys = boost::filesystem;

LabelReaderFileList::LabelReaderFileList() {
    _src_dir = nullptr;
    _entity = nullptr;
    _sub_dir = nullptr;
}

void LabelReaderFileList::init(const MetaDataConfig& cfg, pMetaDataBatch meta_data_batch) {
    _file_list_path = cfg.path();
    _output = meta_data_batch;
}

bool LabelReaderFileList::exists(const std::string& image_name) {
    return _map_content.find(image_name) != _map_content.end();
}

void LabelReaderFileList::add(std::string image_name, int label) {
    pMetaData info = std::make_shared<Label>(label);
    if(exists(image_name)) {
        WRN("Entity with the same name exists")
        return;
    }
    _map_content.insert(pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

void LabelReaderFileList::print_map_contents() {
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_labels()[0] << std::endl;
    }
}

void LabelReaderFileList::release() {
    _map_content.clear();
}

void LabelReaderFileList::release(std::string image_name) {
    if(!exists(image_name)) {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void LabelReaderFileList::lookup(const std::vector<std::string>& image_names) {
    if(image_names.empty()) {
        WRN("No image names passed")
        return;
    }
    if(image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());
    for(unsigned i = 0; i < image_names.size(); i++) {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if(_map_content.end() == it)
            THROW(" Label Reader File List ERROR: Given name not present in the map "+ image_name )
        _output->get_labels_batch()[i] = it->second->get_labels();
    }
}

void LabelReaderFileList::read_all(const std::string& _path) {
    std::string _folder_path = _path;
    if ((_sub_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _folder_path);
    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;
    while((_entity = readdir (_sub_dir)) != nullptr) {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());
    closedir(_sub_dir);
    std::ifstream infile(_file_list_path);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string file_name;
        uint file_label;
        if (!(iss >> file_name >> file_label)) { break; }
        // process pair (file_name, label)
        auto _last_id= file_name;
        auto last_slash_idx = _last_id.find_last_of("\\/");
        if (std::string::npos != last_slash_idx) {
            _last_id.erase(0, last_slash_idx + 1);
        }
        add( _last_id, file_label);
    }
}
