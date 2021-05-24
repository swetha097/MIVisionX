/*
Copyright (c) 2019 - 2020 Advanced Micro Devices, Inc. All rights reserved.

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
#include <utility>
#include <algorithm>
#include <boost/filesystem.hpp>
#include "commons.h"
#include "exception.h"
#include "video_label_reader_folders.h"
#include "video_properties.h"


using namespace std;

namespace filesys = boost::filesystem;

VideoLabelReaderFolders::VideoLabelReaderFolders()
{
    _src_dir = nullptr;
    _entity = nullptr;
    _sub_dir = nullptr;
}

void VideoLabelReaderFolders::init(const MetaDataConfig& cfg)
{
    _path = cfg.path();
    _output = new LabelBatch();
}
bool VideoLabelReaderFolders::exists(const std::string& image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}


void VideoLabelReaderFolders::substring_extraction(std::string const &str, const char delim,  std::vector<std::string> &out)
{
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}


void VideoLabelReaderFolders::add(std::string image_name, int label, int video_frame_count, int start_frame)
{

    // check for mp4
    // String before mp4 string is label
    // populate the filename_framenumber.jpg label
    std::vector<unsigned> video_prop;
    video_prop = open_video_context(image_name.c_str());
    size_t frame_count = video_frame_count ? video_frame_count : video_prop[2];
    if ( video_frame_count + start_frame > video_prop[2] )
        THROW("The given frame numbers in txt file exceeds the maximum frames in the video" + image_name) 
    // std::cerr<<"\n video frame count:: "<<frame_count;
    std::vector<std::string> substrings;
    char delim = '/';
    substring_extraction(image_name, delim, substrings);
    std::string file_name = substrings[substrings.size()- 1];
    for(uint i = start_frame; i < (start_frame + frame_count); i++)
    {
        pMetaData info = std::make_shared<Label>(label);
        // std::cerr<<"\n label:: "<<label;
        std::string frame_name = std::to_string(_video_idx) + "#" + file_name +"_"+  std::to_string(i);
        // std::cerr<<"\n frame_name ::"<<frame_name;
        if(exists(frame_name))
        {
            WRN("Entity with the same name exists")
            return;
        }
        _map_content.insert(pair<std::string, std::shared_ptr<Label>>(frame_name, info));
    }
    _video_idx++;
}

void VideoLabelReaderFolders::print_map_contents()
{
    std::cerr << "\nMap contents: \n";
    for (auto& elem : _map_content) {
        std::cerr << "Name :\t " << elem.first << "\t ID:  " << elem.second->get_label() << std::endl;
    }
}

void VideoLabelReaderFolders::release()
{
    _map_content.clear();
}

void VideoLabelReaderFolders::release(std::string image_name)
{
    if(!exists(image_name))
    {
        WRN("ERROR: Given not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void VideoLabelReaderFolders::lookup(const std::vector<std::string>& image_names)
{
    if(image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if(image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    for(unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        // std::cerr<<"\n lookup image_name :: "<<image_name;
        auto it = _map_content.find(image_name);
        if(_map_content.end() == it)
            THROW("ERROR: Video label reader folders Given name not present in the map"+ image_name )
        _output->get_label_batch()[i] = it->second->get_label();
    }
}

void VideoLabelReaderFolders::read_text_file(const std::string& _path)
{
    std::ifstream text_file(_path);

    if (text_file.good()) {
        //_text_file.open(path.c_str(), std::ifstream::in);
        std::string line;
        int label;
        int start, end;
        float start_time, end_time;
        std::string video_file_name;
        std::vector<unsigned> props;
        while (std::getline(text_file, line)) {
            start = end = 0;
            std::istringstream line_ss(line);
            if (!(line_ss >> video_file_name >> label))
                continue;
            props = open_video_context(video_file_name.c_str());
            if(_enable_timestamps) {
                if (line_ss >> start_time) {
                    if (line_ss >> end_time) {
                        if (start_time >= end_time) {
                            std::cerr << "[WRN] Start and end time/frame are not satisfying the condition, skipping the file " << video_file_name << "\n";
                            continue;
                        }
                        // std::cerr << "Start time : " << start_time << " : " << end_time << "\n";
                        start = start_time * props[3];
                        end = end_time * props[3];
                        end = end ?end : props[2];
                        // std::cerr << start<< " : " << end<< "\n";
                    }
                }
            }
            else {
                if (line_ss >> start) {
                    if (line_ss >> end) {
                        if (start >= end) {
                            std::cerr << "[WRN] Start and end time/frame are the same, skipping the file " << video_file_name << "\n";
                            continue;
                        }
                        end = end ?end : props[2];
                    }
                }
            }
            add(video_file_name, label, (end - start), start);
        }
    }
    else {
        THROW("Can't open the metadata file at " + std::string(_path))
    }
}

void VideoLabelReaderFolders::read_all(const std::string& _path)
{
    std::string _folder_path = _path;

    filesys::path pathObj(_folder_path);

    if (filesys::exists(pathObj) && filesys::is_regular_file(pathObj)) { // Single file as input
        if (pathObj.has_extension() && pathObj.extension().string() == ".txt") {
            read_text_file(_path);
        }
        else if (pathObj.has_extension() && pathObj.extension().string() == ".mp4") {
            add(_path, 0);
        }
    }
    else {
        if ((_sub_dir = opendir(_folder_path.c_str())) == nullptr)
            THROW("ERROR: Failed opening the directory at " + _folder_path);

        std::vector<std::string> entry_name_list;
        std::string _full_path = _folder_path;

        while((_entity = readdir (_sub_dir)) != nullptr)
        {
            std::string entry_name(_entity->d_name);
            if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
            entry_name_list.push_back(entry_name);
            // std::cerr << "\nEntry name : " << _entity->d_name;
        }
        std::sort(entry_name_list.begin(), entry_name_list.end());
        closedir(_sub_dir);

        for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
            std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
            filesys::path pathObj(subfolder_path);
            if(filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
            {
                // ignore files with extensions .tar, .zip, .7z
                auto file_extension_idx = subfolder_path.find_last_of(".");
                if (file_extension_idx  != std::string::npos) {
                    std::string file_extension = subfolder_path.substr(file_extension_idx+1);
                    if ((file_extension == "tar") || (file_extension == "zip") || (file_extension == "7z") || (file_extension == "rar"))
                        continue;
                }
                read_files(_folder_path);
                for(unsigned i = 0; i < _subfolder_video_file_names.size(); i++) {
                    std::cerr<<"\n subfolder_file_name files-> "<< i << ":: "<<_subfolder_video_file_names[i];
                    add(_subfolder_video_file_names[i], i);
                }
                break;  // assume directory has only files.
            }
            else if(filesys::exists(pathObj) && filesys::is_directory(pathObj))
            {
                _folder_path = subfolder_path;
                _subfolder_video_file_names.clear();
                read_files(_folder_path);
                for(unsigned i = 0; i < _subfolder_video_file_names.size(); i++) {
                    std::cerr<<"\n subfolder_file_name directory -> :: "<< i << ":: "<<_subfolder_video_file_names[i];
                    std::vector<std::string> substrings;
                    char delim = '/';
                    substring_extraction(_subfolder_video_file_names[i], delim, substrings);
                    int label = atoi(substrings[substrings.size()- 2].c_str());
                    std::cerr << "The label : " << label << "\n";
                    add(_subfolder_video_file_names[i], label);
                }
            }
        }
    }
    // print_map_contents();
}

void VideoLabelReaderFolders::read_files(const std::string& _path)
{
    if ((_src_dir = opendir (_path.c_str())) == nullptr)
        THROW("ERROR: Failed opening the directory at " + _path);

    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;

        std::string file_path = _path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _file_names.push_back(file_path);
        //_subfolder_video_file_names.push_back(_entity->d_name);
        _subfolder_video_file_names.push_back(file_path);
        std::cerr << "\nRead files : " << _entity->d_name;
        std::cerr << "\nfile path : " << file_path;
    }
    if(_file_names.empty())
        WRN("LabelReader: Could not find any file in " + _path)
    closedir(_src_dir);
    std::sort(_subfolder_video_file_names.begin(), _subfolder_video_file_names.end());
}

