/*
Copyright (c) 2019 - 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <sstream>
#include <iostream>
#include <fstream>
#include <utility>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "commons.h"
#include "exception.h"
#include "text_file_meta_data_reader.h"

using namespace std;

namespace filesys = boost::filesystem;

TextFileMetaDataReader::TextFileMetaDataReader()
{
    _src_dir = nullptr;
    _entity = nullptr;
    _sub_dir = nullptr;
}

void TextFileMetaDataReader::init(const MetaDataConfig &cfg) {
    _file_list_path = cfg.path();
    std::cerr << "\nPath inside text file reader : " << _path;
    _output = new LabelBatch();
}

bool TextFileMetaDataReader::exists(const std::string& image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void TextFileMetaDataReader::add(std::string image_name, int label)
{
    pMetaData info = std::make_shared<Label>(label);
    if(exists(image_name))
    {
        WRN("Entity with the same name exists")
        return;
    }
    std::cerr << "\ncomes inside text file meta data reader : add " << std::endl;
    std::cerr << "\n Image name : " << image_name  << " \t Label " << label << std::endl;
    _map_content.insert(std::pair<std::string, std::shared_ptr<Label>>(image_name, info));
}

/*void TextFileMetaDataReader::print_map_contents()
{
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_label() << std::endl;
    }
}*/

void TextFileMetaDataReader::lookup(const std::vector<std::string> &image_names) {
    //std::cerr << "\n Printing Map Contents";
    //print_map_contents();
    //std::cerr << "\n End of Map Contents";
	if(image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if(image_names.size() != (unsigned)_output->size())   
        _output->resize(image_names.size());
    _output->reset_objects_count();
    for(unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if(_map_content.end() == it)
            THROW("ERROR: Given name not present in the map"+ image_name )
        _output->get_label_batch()[i] = it->second->get_label();
        _output->increment_object_count(it->second->get_object_count());
    }
}

void TextFileMetaDataReader::read_all(const std::string &_path) {
	/*std::ifstream text_file(path.c_str());
	if(text_file.good())
	{
		//_text_file.open(path.c_str(), std::ifstream::in);
		std::string line;
		while(std::getline(text_file, line))
		{
            std::istringstream line_ss(line);
            int label;
            std::string image_name;
            if(!(line_ss>>image_name>>label))
                continue;
			add(image_name, label);
		}
	}
	else
    {
	    THROW("Can't open the metadata file at "+ path)
    }*/
    std::string _folder_path = _path;
    if ((_sub_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while((_entity = readdir (_sub_dir)) != nullptr)
    {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    std::sort(entry_name_list.begin(), entry_name_list.end());
    closedir(_sub_dir);
    uint label_counter = 0;
    std::ifstream infile(_file_list_path);
    std::string line;
    while (std::getline(infile, line))
    {
        std::istringstream iss(line);
        std::string file_name;
        uint file_label;
        if (!(iss >> file_name >> file_label)) { break; } // error

        // process pair (a,b)
        std::cerr<<" \nPrinting the File names & Labels :";
        std::cerr<<" \nFile Name "<< _full_path + "/" + file_name;
        auto _last_id= file_name;
        auto last_slash_idx = _last_id.find_last_of("\\/");
        if (std::string::npos != last_slash_idx)
        {
            _last_id.erase(0, last_slash_idx + 1);
        }
        std::cerr<<" \nFile ID "<< _last_id ;
        std::cerr<<" \nLabel "<< file_label<< std::endl;

        // std::exit(0);
        // add( _full_path + "/" + file_name, file_label);
        add( _last_id, file_label);
        
    }
}

void TextFileMetaDataReader::release(std::string image_name) {
	if(!exists(image_name))
    {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void TextFileMetaDataReader::release() {
	_map_content.clear();
}
