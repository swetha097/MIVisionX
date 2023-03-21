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
#include "maskApi.h"

using namespace std;

void COCOMetaDataReader::init(const MetaDataConfig &cfg)
{
    _path = cfg.path();
    _polygon_mask = cfg.polygon_mask();
    _pixelwise_mask = cfg.pixelwise_mask();
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
        if (_polygon_mask)
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
        std::cout << vertices_count[0][0] << std::endl;
        if (vertices_count[0].size() == 2) {
            std::cout << vertices_count[0][1] << std::endl;
        }
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
        if (_polygon_mask)
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

/*
std::vector<int> COCOMetaDataReader::generate_pixelwise_mask(RLE *R) {
  auto pol = loader_impl.polygons(image_idx);
  auto ver = loader_impl.vertices(image_idx);
  auto masks_info = loader_impl.pixelwise_masks_info(image_idx);
  int h = masks_info.shape[0];
  int w = masks_info.shape[1];
  auto bboxes = loader_impl.bboxes(image_idx);
  auto labels_span = loader_impl.labels(image_idx);
  std::set<int> labels(labels_span.data(),
                       labels_span.data() + labels_span.size());
  if (!labels.size()) {
    return;
  }

  // Create a run-length encoding for each polygon, indexed by label :
  std::map<int, std::vector<RLE> > frPoly;
  std::vector<double> in;
  for (uint polygon_idx = 0; polygon_idx < pol.size(); polygon_idx++) {
    auto &polygon = pol[polygon_idx];
    int mask_idx = polygon[0];
    int start_idx = polygon[1];
    int end_idx = polygon[2];
    assert(mask_idx < labels_span.size());
    int label = labels_span[mask_idx];
    // Convert polygon to encoded mask
    int nver = end_idx - start_idx;
    auto pol_ver = span<const vec2>{ver.data() + start_idx, nver};
    in.resize(pol_ver.size() * 2);
    for (int i = 0, k = 0; i < pol_ver.size(); i++) {
      in[k++] = static_cast<double>(pol_ver[i].x);
      in[k++] = static_cast<double>(pol_ver[i].y);
    }
    RLE M;
    rleInit(&M, 0, 0, 0, 0);
    rleFrPoly(&M, in.data(), pol_ver.size(), h, w);
    frPoly[label].push_back(M);
  }

  // Reserve run-length encodings by labels
  RLE* R;
  rlesInit(&R, *labels.rbegin() + 1);

  // Mask was originally described in RLE format
  for (uint ann_id = 0 ; ann_id < masks_info.mask_indices.size(); ann_id++) {
    const auto &rle = masks_info.rles[ann_id];
    auto mask_idx = masks_info.mask_indices[ann_id];
    int label = labels_span[mask_idx];
    rleInit(&R[label], (*rle)->h, (*rle)->w, (*rle)->m, (*rle)->cnts);
  }

  // Merge each label (from multi-polygons annotations)
  uint lab_cnt = 0;
  for (const auto &rles : frPoly)
    rleMerge(rles.second.data(), &R[rles.first], rles.second.size(), 0);

  // Merge all the labels into a pair of vectors :
  // [2,2,2],[A,B,C] for [A,A,B,B,C,C]
  struct Encoding {
    uint m;
    std::unique_ptr<uint[]> cnts;
    std::unique_ptr<int[]> vals;
  };
  Encoding A;
  A.cnts = std::make_unique<uint[]>(h * w + 1);  // upper-bound
  A.vals = std::make_unique<int[]>(h * w + 1);

  // first copy the content of the first label to the output
  bool v = false;
  A.m = R[*labels.begin()].m;
  for (siz a = 0; a < R[*labels.begin()].m; a++) {
    A.cnts[a] = R[*labels.begin()].cnts[a];
    A.vals[a] = v ? *labels.begin() : 0;
    v = !v;
  }

  // then merge the other labels
  std::unique_ptr<uint[]> cnts = std::make_unique<uint[]>(h * w + 1);
  std::unique_ptr<int[]> vals = std::make_unique<int[]>(h * w + 1);
  for (auto label = ++labels.begin(); label != labels.end(); label++) {
    RLE B = R[*label];
    if (B.cnts == 0)
      continue;

    uint cnt_a = A.cnts[0];
    uint cnt_b = B.cnts[0];
    int next_val_a = A.vals[0];
    int val_a = next_val_a;
    int val_b = *label;
    bool next_vb = false;
    bool vb = next_vb;
    uint nb_seq_a, nb_seq_b;
    nb_seq_a = nb_seq_b = 1;
    int m = 0;

    int cnt_tot = 1;  // check if we advanced at all
    while (cnt_tot > 0) {
      uint c = std::min(cnt_a, cnt_b);
      cnt_tot = 0;
      // advance A
      cnt_a -= c;
      if (!cnt_a && nb_seq_a < A.m) {
        cnt_a = A.cnts[nb_seq_a];  // next sequence for A
        next_val_a = A.vals[nb_seq_a];
        nb_seq_a++;
      }
      cnt_tot += cnt_a;
      // advance B
      cnt_b -= c;
      if (!cnt_b && nb_seq_b < B.m) {
        cnt_b = B.cnts[nb_seq_b++];  // next sequence for B
        next_vb = !next_vb;
      }
      cnt_tot += cnt_b;

      if (val_a && vb)  // there's already a class at this pixel
                        // in this case, the last annotation wins (it's undefined by the spec)
        vals[m] = (!cnt_a) ? val_a : val_b;
      else if (val_a)
        vals[m] = val_a;
      else if (vb)
        vals[m] = val_b;
      else
        vals[m] = 0;
      cnts[m] = c;
      m++;

      // since we switched sequence for A or B, apply the new value from now on
      val_a = next_val_a;
      vb = next_vb;

      if (cnt_a == 0) break;
    }
    // copy back the buffers to the destination encoding
    A.m = m;
    for (int i = 0; i < m; i++) A.cnts[i] = cnts[i];
    for (int i = 0; i < m; i++) A.vals[i] = vals[i];
  }

  // Decode final pixelwise masks encoded via RLE
  memset(mask, 0, h * w * sizeof(int));
  int x = 0, y = 0;
  for (uint i = 0; i < A.m; i++)
    for (uint j = 0; j < A.cnts[i]; j++) {
      mask[x + y * w] = A.vals[i];
      if (++y >= h) {
        y = 0;
        x++;
      }
    }

  // Destroy RLEs
  rlesFree(&R, *labels.rbegin() + 1);
  for (auto rles : frPoly)
    for (auto &rle : rles.second)
      rleFree(&rle);
} */

void COCOMetaDataReader::read_all(const std::string &path)
{
    std::string rle_str;
    std::vector<uint32_t> rle_uints;
    int push_count = 0;
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
                        std::cout << "width:" << img_size.w << std::endl;
                    }
                    else if (0 == std::strcmp(internal_key, "height"))
                    {
                        img_size.h = parser.GetInt();
                        std::cout << "height:" << img_size.h << std::endl;
                    }
                    else if (0 == std::strcmp(internal_key, "file_name"))
                    {
                        image_name = parser.GetString();
                    }
                    else
                    {
                        parser.SkipValue();
                    }
                }
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
                std::array<float, 4> bbox = {};
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
                        std::cout << "Image ID:" << id << std::endl;
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
                    else if (_polygon_mask && 0 == std::strcmp(internal_key, "segmentation"))
                    {
                        RLE *R = (RLE*) malloc(sizeof(RLE));
                        if (parser.PeekType() == kObjectType)
                        {
                            parser.EnterObject();
                            rle_str.clear();
                            rle_uints.clear();
                            int h = -1, w = -1;
                            while (const char* another_key = parser.NextObjectKey()) {
                                if (0 == std::strcmp(another_key, "size")) {
                                RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
                                parser.EnterArray();
                                parser.NextArrayValue();
                                h = parser.GetInt();
                                parser.NextArrayValue();
                                w = parser.GetInt();
                                parser.NextArrayValue();
                                } else if (0 == std::strcmp(another_key, "counts")) {
                                if (parser.PeekType() == kStringType) {
                                    rle_str = parser.GetString();
                                } else if (parser.PeekType() == kArrayType) {
                                    parser.EnterArray();
                                    while (parser.NextArrayValue()) {
                                    rle_uints.push_back(parser.GetInt());
                                    }
                                } else {
                                    parser.SkipValue();
                                }
                                } else {
                                parser.SkipValue();
                                }
                            }
                            std::cout << "Enters here 2222" << std::endl;
                            if (!rle_str.empty()) {
                                rleInit(R, h, w, rle_uints.size(), const_cast<uint*>(rle_uints.data()));
                            } else if (!rle_uints.empty()) {
                                rleFrString(R, const_cast<char*>(rle_str.c_str()), h, w);
                            }

                            std::cout << "Enters here 22221" << std::endl;
                            std::cout << R->h << " " << R->w << " " << R->m << std::endl;
                            free(R);
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
                char buffer[13];
                sprintf(buffer, "%012d", id);
                string str(buffer);
                std::string file_name = str + ".jpg";

                auto it = _map_img_sizes.find(file_name);
                ImgSize image_size = it->second; //Normalizing the co-ordinates & convert to "ltrb" format
                if (_polygon_mask && iscrowd == 0)
                {
                    std::cout << "Enters here 111" << push_count++ << "\t" << file_name << std::endl;
                    box.l = bbox[0] / image_size.w;
                    box.t = bbox[1] / image_size.h;
                    box.r = (bbox[0] + bbox[2] - 1) / image_size.w;
                    box.b = (bbox[1] + bbox[3] - 1) / image_size.h;
                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    polygon_count.push_back(polygon_size);
                    vertices_count.push_back(vertices_array);
                    add(file_name, bb_coords, bb_labels, image_size, mask, polygon_count, vertices_count);
                    mask.clear();
                    polygon_size = 0;
                    polygon_count.clear();
                    vertices_count.clear();
                    vertices_array.clear();
                    bb_coords.clear();
                    bb_labels.clear();
                }
                else if (!_polygon_mask)
                {
                    box.l = bbox[0] / image_size.w;
                    box.t = bbox[1] / image_size.h;
                    box.r = (bbox[0] + bbox[2]) / image_size.w;
                    box.b = (bbox[1] + bbox[3]) / image_size.h;
                    bb_coords.push_back(box);
                    bb_labels.push_back(label);
                    add(file_name, bb_coords, bb_labels, image_size);
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
            int cnt_idx = _it_label->second;
            continuous_label_id.push_back(cnt_idx);
        }
        elem.second->set_bb_labels(continuous_label_id);
    }
    _coco_metadata_read_time.end(); // Debug timing
    //print_map_contents();
    // std::cout << "coco read time in sec: " << _coco_metadata_read_time.get_timing() / 1000 << std::endl;
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
