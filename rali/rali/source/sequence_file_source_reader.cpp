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

#include <cassert>
#include <algorithm>
#include <commons.h>
#include "sequence_file_source_reader.h"
#include <boost/filesystem.hpp>

namespace filesys = boost::filesystem;

SequenceFileSourceReader::SequenceFileSourceReader():
_shuffle_time("shuffle_time", DBG_TIMING)
{
    _src_dir = nullptr;
    _sub_dir = nullptr;
    _entity = nullptr;
    _curr_file_idx = 0;
    _current_file_size = 0;
    _current_fPtr = nullptr;
    _loop = false;
    _file_id = 0;
    _shuffle = false;
    _file_count_all_shards = 0;
}

unsigned SequenceFileSourceReader::count()
{
    if(_loop)
        return _sequence_frame_names.size();

    int ret = ((int)_sequence_frame_names.size() -_read_counter);
    return ((ret < 0) ? 0 : ret);
}

Reader::Status SequenceFileSourceReader::initialize(ReaderConfig desc)
{
    auto ret = Reader::Status::OK;
    _file_id = 0;
    _folder_path = desc.path();
    _shard_id = desc.get_shard_id();
    _shard_count = desc.get_shard_count();
    _batch_count = desc.get_batch_size();
    _shuffle = desc.shuffle();
    _loop = desc.loop();
    _sequence_length =  desc.get_sequence_length();
    std::cerr << "\nSequence_length : " << _sequence_length;

    ret = subfolder_reading();
    for(unsigned i = 0 ; i < _video_file_names.size(); i++)
    {
        if(_sequence_length > _video_file_names[i].size())
        {
            THROW("\nSequence length is not valid");
        }
        for(unsigned j = 0; (j+_sequence_length) <= _video_file_names[i].size(); j++)
        {
            std::cerr << "\n";
            std::vector< std::string> temp;
            for(unsigned x=0, k = j; x < _sequence_length; x++, k++)
            {                   
                std::cerr << "\t "<< _video_file_names[i][k];
                temp.push_back(_video_file_names[i][k]);
            }
            _sequence_frame_names.push_back(temp);
        }
    }

    for (int i = 0; i < _sequence_frame_names.size(); i++)
    {
        std::cerr << "\n Id: " << i << std::endl;
        for (int j = 0; j < _sequence_frame_names[i].size(); j++) {
            std::cerr << "\t Video files : " << _sequence_frame_names[i][j] << std::endl;
        }
    }
    //exit(0);

    // the following code is required to make every shard the same size:: required for multi-gpu training
    /*if (_shard_count > 1 && _batch_count > 1) {
        int _num_batches = _file_names.size()/_batch_count;
        int max_batches_per_shard = (_file_count_all_shards + _shard_count-1)/_shard_count;
        max_batches_per_shard = (max_batches_per_shard + _batch_count-1)/_batch_count;
        if (_num_batches < max_batches_per_shard) {
            replicate_last_batch_to_pad_partial_shard();
        }
    }*/
    //shuffle dataset if set
    _shuffle_time.start();
    if( ret==Reader::Status::OK && _shuffle)
        std::random_shuffle(_sequence_frame_names.begin(), _sequence_frame_names.end());
    _shuffle_time.end();
    return ret;

}

void SequenceFileSourceReader::incremenet_read_ptr()
{
    //_read_counter++;
    //_curr_file_idx = (_curr_file_idx + 1) % _file_names.size();
    _read_counter += _sequence_length;
    _curr_file_idx = (_curr_file_idx + 1) % _sequence_frame_names.size();
}
size_t SequenceFileSourceReader::open()
{
    auto file_path = _sequence_frame_names[_curr_file_idx][0];// Get next file name
    incremenet_read_ptr();
    _last_id= file_path;
    auto last_slash_idx = _last_id.find_last_of("\\/");
    if (std::string::npos != last_slash_idx)
    {
        _last_id.erase(0, last_slash_idx + 1);
    }

    _current_fPtr = fopen(file_path.c_str(), "rb");// Open the file,

    if(!_current_fPtr) // Check if it is ready for reading
        return 0;

    fseek(_current_fPtr, 0 , SEEK_END);// Take the file read pointer to the end

    _current_file_size = ftell(_current_fPtr);// Check how many bytes are there between and the current read pointer position (end of the file)

    if(_current_file_size == 0)
    { // If file is empty continue
        fclose(_current_fPtr);
        _current_fPtr = nullptr;
        return 0;
    }

    fseek(_current_fPtr, 0 , SEEK_SET);// Take the file pointer back to the start

    return _current_file_size;
}

size_t SequenceFileSourceReader::read(unsigned char* buf, size_t read_size)
{
    if(!_current_fPtr)
        return 0;

    // Requested read size bigger than the file size? just read as many bytes as the file size
    read_size = (read_size > _current_file_size) ? _current_file_size : read_size;

    size_t actual_read_size = fread(buf, sizeof(unsigned char), read_size, _current_fPtr);
    return actual_read_size;
}

int SequenceFileSourceReader::close()
{
    return release();
}

SequenceFileSourceReader::~SequenceFileSourceReader()
{
    release();
}

int
SequenceFileSourceReader::release()
{
    if(!_current_fPtr)
        return 0;
    fclose(_current_fPtr);
    _current_fPtr = nullptr;
    return 0;
}

void SequenceFileSourceReader::reset()
{
    _shuffle_time.start();
    if (_shuffle) std::random_shuffle(_sequence_frame_names.begin(), _sequence_frame_names.end());
    _shuffle_time.end();
    _read_counter = 0;
    _curr_file_idx = 0;
}

Reader::Status SequenceFileSourceReader::subfolder_reading()
{
    if ((_sub_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);

    std::vector<std::string> entry_name_list;
    std::string _full_path = _folder_path;

    while((_entity = readdir (_sub_dir)) != nullptr)
    {
        std::string entry_name(_entity->d_name);
        if (strcmp(_entity->d_name, ".") == 0 || strcmp(_entity->d_name, "..") == 0) continue;
        entry_name_list.push_back(entry_name);
    }
    closedir(_sub_dir);
    std::sort(entry_name_list.begin(), entry_name_list.end());

    auto ret = Reader::Status::OK;
    for (unsigned dir_count = 0; dir_count < entry_name_list.size(); ++dir_count) {
        std::string subfolder_path = _full_path + "/" + entry_name_list[dir_count];
        std::cerr << "\n Subfoler path : " << subfolder_path;
        filesys::path pathObj(subfolder_path);
        if(filesys::exists(pathObj) && filesys::is_regular_file(pathObj))
        {
            break;  // assume directory has only files.
        }
        else if(filesys::exists(pathObj) && filesys::is_directory(pathObj))
        {
            _folder_path = subfolder_path;
            if(open_folder() != Reader::Status::OK)
                WRN("FileReader ShardID ["+ TOSTR(_shard_id)+ "] File reader cannot access the storage at " + _folder_path);
            _video_file_names.push_back(_file_names);
            _file_names.clear();
        }
    }
    /*if(_in_batch_read_count > 0 && _in_batch_read_count < _batch_count)
    {
        replicate_last_image_to_fill_last_shard();
        LOG("FileReader ShardID [" + TOSTR(_shard_id) + "] Replicated " + _folder_path+_last_file_name + " " + TOSTR((_batch_count - _in_batch_read_count) ) + " times to fill the last batch")
    }
    if(!_file_names.empty())
        LOG("FileReader ShardID ["+ TOSTR(_shard_id)+ "] Total of " + TOSTR(_file_names.size()) + " images loaded from " + _full_path )*/
    return ret;
}
void SequenceFileSourceReader::replicate_last_image_to_fill_last_shard()
{
    for(size_t i = _in_batch_read_count; i < _batch_count; i++)
        _file_names.push_back(_last_file_name);
}

void SequenceFileSourceReader::replicate_last_batch_to_pad_partial_shard()
{
    if (_file_names.size() >=  _batch_count) {
        for (size_t i = 0; i < _batch_count; i++)
            _file_names.push_back(_file_names[i - _batch_count]);
    }
}

Reader::Status SequenceFileSourceReader::open_folder()
{
    if ((_src_dir = opendir (_folder_path.c_str())) == nullptr)
        THROW("FileReader ShardID ["+ TOSTR(_shard_id)+ "] ERROR: Failed opening the directory at " + _folder_path);


    while((_entity = readdir (_src_dir)) != nullptr)
    {
        if(_entity->d_type != DT_REG)
            continue;

        if(get_file_shard_id() != _shard_id )
        {
            _file_count_all_shards++;
            incremenet_file_id();
            continue;
        }
        _in_batch_read_count++;
        _in_batch_read_count = (_in_batch_read_count%_batch_count == 0) ? 0 : _in_batch_read_count;
        std::string file_path = _folder_path;
        file_path.append("/");
        file_path.append(_entity->d_name);
        _last_file_name = file_path;
        _file_names.push_back(file_path);
        _file_count_all_shards++;
        incremenet_file_id();
    }
    if(_file_names.empty())
        WRN("FileReader ShardID ["+ TOSTR(_shard_id)+ "] Did not load any file from " + _folder_path)

    closedir(_src_dir);
    return Reader::Status::OK;
}

size_t SequenceFileSourceReader::get_file_shard_id()
{
    if(_batch_count == 0 || _shard_count == 0)
        THROW("Shard (Batch) size cannot be set to 0")
    //return (_file_id / (_batch_count)) % _shard_count;
    return _file_id  % _shard_count;
}
