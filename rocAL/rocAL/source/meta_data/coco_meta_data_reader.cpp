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

#include "coco_meta_data_reader.h"
#include <iostream>
#include <utility>
#include <algorithm>
#include <fstream>
#include "lookahead_parser.h"

using namespace std;

void COCOMetaDataReader::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _mask = cfg.mask();
    _output = new BoundingBoxBatch();
}

bool COCOMetaDataReader::exists(const std::string &image_name)
{
    return _map_content.find(image_name) != _map_content.end();
}

void COCOMetaDataReader::lookup(const std::vector<std::string> &image_names)
{

    if (image_names.empty())
    {
        WRN("No image names passed")
        return;
    }
    if (image_names.size() != (unsigned)_output->size())
        _output->resize(image_names.size());

    _output->reset_objects_count();
    for (unsigned i = 0; i < image_names.size(); i++)
    {
        auto image_name = image_names[i];
        auto it = _map_content.find(image_name);
        if (_map_content.end() == it)
            THROW("ERROR: Given name not present in the map" + image_name)
        _output->get_bb_cords_batch()[i] = it->second->get_bb_cords();
        _output->get_bb_labels_batch()[i] = it->second->get_bb_labels();
        _output->get_img_sizes_batch()[i] = it->second->get_img_size();
        _output->increment_object_count(it->second->get_object_count());
        _output->get_metadata_dimensions_batch().bb_labels_dims()[i] = it->second->get_bb_label_dims();
        _output->get_metadata_dimensions_batch().bb_cords_dims()[i] = it->second->get_bb_cords_dims();
        if (_mask)
        {
            _output->get_mask_cords_batch()[i] = it->second->get_mask_cords();
            _output->get_mask_polygons_count_batch()[i] = it->second->get_polygon_count();
            _output->get_mask_vertices_count_batch()[i] = it->second->get_vertices_count();
            _output->get_metadata_dimensions_batch().mask_cords_dims()[i] = it->second->get_mask_cords_dims();
            _output->increment_mask_coords_count(it->second->get_mask_coords_count());
        }
    }
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, BoundingBoxLabels bb_labels, ImgSize image_size, MaskCords mask_cords, std::vector<int> polygon_count, std::vector<std::vector<int>> vertices_count)
{
    if (exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_bb_labels().push_back(bb_labels[0]);
        it->second->get_mask_cords().insert(it->second->get_mask_cords().end(), mask_cords.begin(), mask_cords.end());
        it->second->get_polygon_count().push_back(polygon_count[0]);
        it->second->get_vertices_count().push_back(vertices_count[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size, mask_cords, polygon_count, vertices_count);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void COCOMetaDataReader::add(std::string image_name, BoundingBoxCords bb_coords, BoundingBoxLabels bb_labels, ImgSize image_size)
{
    if (exists(image_name))
    {
        auto it = _map_content.find(image_name);
        it->second->get_bb_cords().push_back(bb_coords[0]);
        it->second->get_bb_labels().push_back(bb_labels[0]);
        return;
    }
    pMetaDataBox info = std::make_shared<BoundingBox>(bb_coords, bb_labels, image_size);
    _map_content.insert(pair<std::string, std::shared_ptr<BoundingBox>>(image_name, info));
}

void COCOMetaDataReader::print_map_contents()
{
    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;
    ImgSize img_size;
    MaskCords mask_cords;
    std::vector<int> polygon_size;
    std::vector<std::vector<int>> vertices_count;

    std::cout << "\nBBox Annotations List: \n";
    for (auto &elem : _map_content)
    {
        std::cout << "\nName :\t " << elem.first;
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_bb_labels();
        img_size = elem.second->get_img_size();
        mask_cords = elem.second->get_mask_cords();
        polygon_size = elem.second->get_polygon_count();
        vertices_count = elem.second->get_vertices_count();
        std::cout << "<wxh, num of bboxes>: " << img_size.w << " X " << img_size.h << " , " << bb_coords.size() << std::endl;
        for (unsigned int i = 0; i < bb_coords.size(); i++)
        {
            std::cout << " l : " << bb_coords[i].l << " t: :" << bb_coords[i].t << " r : " << bb_coords[i].r << " b: :" << bb_coords[i].b << "Label Id : " << bb_labels[i] << std::endl;
        }
        if (_mask)
        {
            int count = 0;
            std::cout << "\nNumber of objects : " << bb_coords.size() << std::endl;
            for (unsigned int i = 0; i < bb_coords.size(); i++)
            {
                std::cout << "\nNumber of polygons for object[ << " << i << "]:" << polygon_size[i];
                for (int j = 0; j < polygon_size[i]; j++)
                {
                    std::cout << "\nPolygon size :" << vertices_count[i][j] << "Elements::";
                    for (int k = 0; k < vertices_count[i][j]; k++, count++)
                        std::cout << "\t " << mask_cords[count + vertices_count[i][j]];
                }
            }
        }
    }
}

void COCOMetaDataReader::read_all(const std::string &path)
{
    _coco_metadata_read_time.start(); // Debug timing
    std::ifstream f;
    f.open (path, std::ifstream::in|std::ios::binary);
    if (f.fail()) THROW("ERROR: Given annotations file not present " + path);
    f.ignore( std::numeric_limits<std::streamsize>::max() );
    auto file_size = f.gcount();
    f.clear();   //  Since ignore will have set eof.
    if (file_size == 0)
    { // If file is empty return
        f.close();
        THROW("ERROR: Given annotations file not valid " + path);
    }
    std::unique_ptr<char, std::function<void(char *)>> buff(
        new char[file_size + 1],
        [](char *data)
        { delete[] data; });
    f.seekg(0, std::ios::beg);
    buff.get()[file_size] = '\0';
    f.read(buff.get(), file_size);
    f.close();

    LookaheadParser parser(buff.get());

    BoundingBoxCords bb_coords;
    BoundingBoxLabels bb_labels;
    ImgSizes img_sizes;
    std::vector<int> polygon_count;
    int polygon_size = 0;
    std::vector<std::vector<int>> vertices_count;

    BoundingBoxCord box;
    ImgSize img_size;
    RAPIDJSON_ASSERT(parser.PeekType() == kObjectType);
    parser.EnterObject();
    while (const char *key = parser.NextObjectKey())
    {
        if (0 == std::strcmp(key, "images"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue())
            {
                string image_name;
                int original_id;
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "width"))
                    {
                        img_size.w = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "height"))
                    {
                        img_size.h = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "file_name"))
                    {
                        image_name = parser.GetString();
                    }
                    else if(0 == std::strcmp(internal_key, "id"))
                    {
                        img_size.original_id = parser.GetInt();
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
                _map_img_names.insert(pair<int, std::string>(img_size.original_id, image_name));
                _map_img_sizes.insert(pair<std::string, ImgSize>(image_name, img_size));
                img_size = {};
            }
        }
        else if (0 == std::strcmp(key, "categories"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            int id = 1, continuous_idx = 1;
            while (parser.NextArrayValue())
            {
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "id"))
                    {
                        id = parser.GetInt();
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
                _label_info.insert(std::make_pair(id, continuous_idx));
                continuous_idx++;
            }
        }
        else if (0 == std::strcmp(key, "annotations"))
        {
            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
            parser.EnterArray();
            while (parser.NextArrayValue())
            {
                int id = 1, label = 0, iscrowd = 0;
                std::array<double, 4> bbox = {};
                std::vector<float> mask;
                std::vector<int> vertices_array;
                if (parser.PeekType() != kObjectType)
                {
                    continue;
                }
                parser.EnterObject();
                while (const char *internal_key = parser.NextObjectKey())
                {
                    if (0 == std::strcmp(internal_key, "image_id"))
                    {
                        id = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "category_id"))
                    {
                        label = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "iscrowd"))
                    {
                        iscrowd = parser.GetInt();
                    }
                    else if (0 == std::strcmp(internal_key, "bbox"))
                    {
                        RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                        parser.EnterArray();
                        int i = 0;
                        while (parser.NextArrayValue())
                        {
                            bbox[i] = parser.GetDouble();
                            ++i;
                        }
                    }
                    else if (_mask && 0 == std::strcmp(internal_key, "segmentation"))
                    {
                        if (parser.PeekType() == kObjectType)
                        {
                            parser.EnterObject();
                            const char *key = parser.NextObjectKey();
                            if (0 == std::strcmp(key, "counts"))
                            {
                                parser.SkipArray();
                            }
                        }
                        else
                        {
                            RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                            parser.EnterArray();
                            while (parser.NextArrayValue())
                            {
                                polygon_size += 1;
                                int vertex_count = 0;
                                parser.EnterArray();
                                while (parser.NextArrayValue())
                                {

                                    mask.push_back(parser.GetDouble());
                                    vertex_count += 1;
                                }
                                vertices_array.push_back(vertex_count);
                            }
                        }
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }

                auto itr = _map_img_names.find(id);
                auto it = _map_img_sizes.find(itr->second);
                ImgSize image_size = it->second; //Normalizing the co-ordinates & convert to "ltrb" format
                if (_mask && iscrowd == 0)
                {
                    box.l = bbox[0] / image_size.w;
                    box.t = bbox[1] / image_size.h;
                    box.r = (bbox[0] + bbox[2] - 1) / image_size.w;
                    box.b = (bbox[1] + bbox[3] - 1) / image_size.h;
                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    polygon_count.push_back(polygon_size);
                    vertices_count.push_back(vertices_array);
                    add(itr->second, bb_coords, bb_labels, image_size, mask, polygon_count, vertices_count);
                    mask.clear();
                    polygon_size = 0;
                    polygon_count.clear();
                    vertices_count.clear();
                    vertices_array.clear();
                    bb_coords.clear();
                    bb_labels.clear();
                }
                else if (!_mask)
                {
                    box.l = bbox[0] / static_cast<double>(image_size.w);
                    box.t = bbox[1] / static_cast<double>(image_size.h);
                    box.r = (bbox[0] + bbox[2]) / static_cast<double>(image_size.w);
                    box.b = (bbox[1] + bbox[3]) / static_cast<double>(image_size.h);
                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    add(itr->second, bb_coords, bb_labels, image_size);
                    bb_coords.clear();
                    bb_labels.clear();
                }
                image_size = {};
            }
        }
        else
        {
            parser.SkipValue();
        }
    }
    for (auto &elem : _map_content)
    {
        bb_coords = elem.second->get_bb_cords();
        bb_labels = elem.second->get_bb_labels();
        BoundingBoxLabels continuous_label_id;
        for (unsigned int i = 0; i < bb_coords.size(); i++)
        {
            auto _it_label = _label_info.find(bb_labels[i]);
            //int cnt_idx = _it_label->second;
            int cnt_idx = _it_label->first;
            continuous_label_id.push_back(cnt_idx);
        }
        elem.second->set_bb_labels(continuous_label_id);
    }
    _coco_metadata_read_time.end(); // Debug timing
    //print_map_contents();
    std::cout << "coco read time in sec: " << _coco_metadata_read_time.get_timing() / 1000 << std::endl;
}

void COCOMetaDataReader::release(std::string image_name)
{
    if (!exists(image_name))
    {
        WRN("ERROR: Given name not present in the map" + image_name);
        return;
    }
    _map_content.erase(image_name);
}

void COCOMetaDataReader::release()
{
    _map_content.clear();
    _map_img_sizes.clear();
}

COCOMetaDataReader::COCOMetaDataReader() : _coco_metadata_read_time("coco meta read time", DBG_TIMING)
{
}
