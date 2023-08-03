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
#include "bounding_box_graph.h"

void BoundingBoxGraph::process(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data)
{
    for (auto &meta_node : _meta_nodes)
    {
        meta_node->update_parameters(input_meta_data, output_meta_data);
    }
}

void BoundingBoxGraph::update_meta_data(pMetaDataBatch input_meta_data, decoded_image_info decode_image_info)
{
    std::vector<uint32_t> original_height = decode_image_info._original_height;
    std::vector<uint32_t> original_width = decode_image_info._original_width;
    std::vector<uint32_t> roi_width = decode_image_info._roi_width;
    std::vector<uint32_t> roi_height = decode_image_info._roi_height;
    for (int i = 0; i < input_meta_data->size(); i++)
    {
        float _dst_to_src_width_ratio = roi_width[i] / static_cast<float>(original_width[i]);
        float _dst_to_src_height_ratio = roi_height[i] / static_cast<float>(original_height[i]);
        unsigned bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf = input_meta_data->get_bb_cords_batch()[i];
        BoundingBoxCords bb_coords;
        BoundingBoxCord temp_box;
        temp_box.l = temp_box.t = temp_box.r = temp_box.b = 0;
        for (uint j = 0; j < bb_count; j++)
        {
            coords_buf[j].l *= _dst_to_src_width_ratio;
            coords_buf[j].t *= _dst_to_src_height_ratio;
            coords_buf[j].r *= _dst_to_src_width_ratio;
            coords_buf[j].b *= _dst_to_src_height_ratio;
            bb_coords.push_back(coords_buf[j]);
        }
        if (bb_coords.size() == 0)
        {
            bb_coords.push_back(temp_box);
        }
        input_meta_data->get_bb_cords_batch()[i] = bb_coords;
    }
}

inline float ssd_BBoxIntersectionOverUnion(const BoundingBoxCord &box1, const float &box1_area, const BoundingBoxCordf &box2)
{
    float xA = std::max(static_cast<float>(box1.l), box2.ltrb.l);
    float yA = std::max(static_cast<float>(box1.t), box2.ltrb.t);
    float xB = std::min(static_cast<float>(box1.r), box2.ltrb.r);
    float yB = std::min(static_cast<float>(box1.b), box2.ltrb.b);
    float intersection_area = std::max((float)0.0, xB - xA) * std::max((float)0.0, yB - yA);
    float box2_area = (box2.ltrb.b - box2.ltrb.t) * (box2.ltrb.r - box2.ltrb.l);
    return (float) (intersection_area / (box1_area + box2_area - intersection_area));
}

void BoundingBoxGraph::update_random_bbox_meta_data(pMetaDataBatch input_meta_data, pMetaDataBatch output_meta_data, decoded_image_info decode_image_info, crop_image_info crop_image_info)
{
    std::vector<uint32_t> original_height = decode_image_info._original_height;
    std::vector<uint32_t> original_width = decode_image_info._original_width;
    std::vector<uint32_t> roi_width = decode_image_info._roi_width;
    std::vector<uint32_t> roi_height = decode_image_info._roi_height;
    auto crop_cords = crop_image_info._crop_image_coords;
    for (int i = 0; i < input_meta_data->size(); i++)
    {
        auto bb_count = input_meta_data->get_labels_batch()[i].size();
        BoundingBoxCords coords_buf;
        Labels labels_buf;
        coords_buf.resize(bb_count);
        labels_buf.resize(bb_count);
        memcpy(labels_buf.data(), input_meta_data->get_labels_batch()[i].data(), sizeof(int) * bb_count);
        memcpy((void *)coords_buf.data(), input_meta_data->get_bb_cords_batch()[i].data(), input_meta_data->get_bb_cords_batch()[i].size() * sizeof(BoundingBoxCord));
        BoundingBoxCords bb_coords;
        Labels bb_labels;
        BoundingBoxCord crop_box;
        crop_box.l = crop_cords[i][0];
        crop_box.t = crop_cords[i][1];
        crop_box.r = crop_box.l + crop_cords[i][2];
        crop_box.b = crop_box.t + crop_cords[i][3];
        float w_factor = 1.0 / crop_cords[i][2];
        float h_factor = 1.0 / crop_cords[i][3];
        for (uint j = 0; j < bb_count; j++)
        {
            //Mask Criteria
            auto x_c = 0.5 * (coords_buf[j].l + coords_buf[j].r);
            auto y_c = 0.5 * (coords_buf[j].t + coords_buf[j].b);
            if ((x_c > crop_box.l) && (x_c < crop_box.r) && (y_c > crop_box.t) && (y_c < crop_box.b))
            {
                float xA = std::max(crop_box.l, coords_buf[j].l);
                float yA = std::max(crop_box.t, coords_buf[j].t);
                float xB = std::min(crop_box.r, coords_buf[j].r);
                float yB = std::min(crop_box.b, coords_buf[j].b);
                coords_buf[j].l = (xA - crop_box.l) * w_factor;
                coords_buf[j].t = (yA - crop_box.t) * h_factor;
                coords_buf[j].r = (xB - crop_box.l) * w_factor;
                coords_buf[j].b = (yB - crop_box.t) * h_factor;
                bb_coords.push_back(coords_buf[j]);
                bb_labels.push_back(labels_buf[j]);
            }
        }
        if (bb_coords.size() == 0)
        {
            THROW("Bounding box co-ordinates not found in the image ");
        }
        output_meta_data->get_bb_cords_batch()[i] = bb_coords;
        output_meta_data->get_labels_batch()[i] = bb_labels;
    }
}

inline void calculate_ious_for_box(float *ious, BoundingBoxCord &box, BoundingBoxCordf *anchors, unsigned int num_anchors)
{
    float box_area = (box.b - box.t) * (box.r - box.l);
    ious[0] = ssd_BBoxIntersectionOverUnion(box, box_area, anchors[0]);

    int best_idx = 0;
    float best_iou = ious[0];
    for (unsigned int anchor_idx = 1; anchor_idx < num_anchors; anchor_idx++)
    {
        ious[anchor_idx] = ssd_BBoxIntersectionOverUnion(box, box_area, anchors[anchor_idx]);
        if (ious[anchor_idx] > best_iou)
        {
            best_iou = ious[anchor_idx];
            best_idx = anchor_idx;
        }
    }
    // For best default box matched with current object let iou = 2, to make sure there is a match,
    // as this object will be the best (highest IoU), for this default box
    ious[best_idx] = 2.;
}

inline int find_best_box_for_anchor(unsigned anchor_idx, const std::vector<float> &ious, unsigned num_boxes, unsigned anchors_size)
{
    unsigned best_idx = 0;
    float best_iou = ious[anchor_idx];
    for (unsigned bbox_idx = 1; bbox_idx < num_boxes; ++bbox_idx)
    {
        if (ious[bbox_idx * anchors_size + anchor_idx] >= best_iou)
        {
            best_iou = ious[bbox_idx * anchors_size + anchor_idx];
            best_idx = bbox_idx;
        }
    }
    return best_idx;
}

void BoundingBoxGraph::update_box_encoder_meta_data(std::vector<float> *anchors, pMetaDataBatch full_batch_meta_data, float criteria, bool offset, float scale, std::vector<float>& means, std::vector<float>& stds, float *encoded_boxes_data, int *encoded_labels_data)
{
    #pragma omp parallel for
    for (int i = 0; i < full_batch_meta_data->size(); i++)
    {
        BoundingBoxCordf *bbox_anchors = reinterpret_cast<BoundingBoxCordf *>(anchors->data());
        auto bb_count = full_batch_meta_data->get_labels_batch()[i].size();
        int* bb_labels = full_batch_meta_data->get_labels_batch()[i].data();
        BoundingBoxCord *bb_coords = reinterpret_cast<BoundingBoxCord *>(full_batch_meta_data->get_bb_cords_batch()[i].data());
        unsigned anchors_size = anchors->size() / 4; // divide the anchors_size by 4 to get the total number of anchors
        int* encoded_labels = encoded_labels_data + (i * anchors_size);
        BoundingBoxCordf* encoded_bb = reinterpret_cast<BoundingBoxCordf *>(encoded_boxes_data + (i * anchors_size * 4));
        //Calculate Ious
        //ious size - bboxes count x anchors count
        std::vector<float> ious(bb_count * anchors_size);
        for (uint bb_idx = 0; bb_idx < bb_count; bb_idx++)
        {
            auto iou_rows = ious.data() + (bb_idx * (anchors_size));
            calculate_ious_for_box(iou_rows, bb_coords[bb_idx], bbox_anchors, anchors_size);
        }
        float inv_stds[4] = {(float)(1./stds[0]), (float)(1./stds[1]), (float)(1./stds[2]), (float)(1./stds[3])};
        float half_scale = 0.5 * scale;
        // Depending on the matches ->place the best bbox instead of the corresponding anchor_idx in anchor
        for (unsigned anchor_idx = 0; anchor_idx < anchors_size; anchor_idx++)
        {
            BoundingBoxCordf box_bestidx, anchor_xcyxwh;
            BoundingBoxCordf *p_anchor = &bbox_anchors[anchor_idx];
            const auto best_idx = find_best_box_for_anchor(anchor_idx, ious, bb_count, anchors_size);
            // Filter matches by criteria
            if (ious[(best_idx * anchors_size) + anchor_idx] > criteria) //Its a match
            {
                //Convert the "ltrb" format to "xcycwh"
                if (offset)
                {
                    box_bestidx.xcycwh.xc = (bb_coords[best_idx].l + bb_coords[best_idx].r) * half_scale; //xc
                    box_bestidx.xcycwh.yc = (bb_coords[best_idx].t + bb_coords[best_idx].b) * half_scale; //yc
                    box_bestidx.xcycwh.w = (bb_coords[best_idx].r - bb_coords[best_idx].l) * scale;      //w
                    box_bestidx.xcycwh.h = (bb_coords[best_idx].b - bb_coords[best_idx].t) * scale;      //h
                    //Convert the "ltrb" format to "xcycwh"
                    anchor_xcyxwh.xcycwh.xc = (p_anchor->ltrb.l + p_anchor->ltrb.r) * half_scale; //xc
                    anchor_xcyxwh.xcycwh.yc = (p_anchor->ltrb.t + p_anchor->ltrb.b) * half_scale; //yc
                    anchor_xcyxwh.xcycwh.w = ( p_anchor->ltrb.r - p_anchor->ltrb.l ) * scale;      //w
                    anchor_xcyxwh.xcycwh.h = ( p_anchor->ltrb.b - p_anchor->ltrb.t ) * scale;      //h
                    // Reference for offset calculation between the Ground Truth bounding boxes & anchor boxes in <xc,yc,w,h> format
                    // https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection#predictions-vis-%C3%A0-vis-priors
                    box_bestidx.xcycwh.xc = ((box_bestidx.xcycwh.xc - anchor_xcyxwh.xcycwh.xc) / anchor_xcyxwh.xcycwh.w - means[0]) * inv_stds[0] ;
                    box_bestidx.xcycwh.yc = ((box_bestidx.xcycwh.yc - anchor_xcyxwh.xcycwh.yc) / anchor_xcyxwh.xcycwh.h - means[1]) * inv_stds[1];
                    box_bestidx.xcycwh.w = (std::log(box_bestidx.xcycwh.w / anchor_xcyxwh.xcycwh.w) - means[2]) * inv_stds[2];
                    box_bestidx.xcycwh.h = (std::log(box_bestidx.xcycwh.h / anchor_xcyxwh.xcycwh.h) - means[3]) * inv_stds[3];
                    encoded_bb[anchor_idx] = box_bestidx;
                    encoded_labels[anchor_idx] = bb_labels[best_idx];
                }
                else
                {
                    box_bestidx.xcycwh.xc = 0.5 * (bb_coords[best_idx].l + bb_coords[best_idx].r); //xc
                    box_bestidx.xcycwh.yc = 0.5 * (bb_coords[best_idx].t + bb_coords[best_idx].b); //yc
                    box_bestidx.xcycwh.w = bb_coords[best_idx].r - bb_coords[best_idx].l;      //w
                    box_bestidx.xcycwh.h = bb_coords[best_idx].b - bb_coords[best_idx].t;      //h
                    encoded_bb[anchor_idx] = box_bestidx;
                    encoded_labels[anchor_idx] = bb_labels[best_idx];
                }
            }
            else // Not a match
            {
                if (offset)
                {
                    encoded_bb[anchor_idx] = {0, 0, 0, 0};
                    encoded_labels[anchor_idx] = 0;
                }
                else
                {
                    //Convert the "ltrb" format to "xcycwh"
                    encoded_bb[anchor_idx].xcycwh.xc = 0.5 * (p_anchor->ltrb.l + p_anchor->ltrb.r); //xc
                    encoded_bb[anchor_idx].xcycwh.yc = 0.5 * (p_anchor->ltrb.t + p_anchor->ltrb.b); //yc
                    encoded_bb[anchor_idx].xcycwh.w = (-p_anchor->ltrb.l + p_anchor->ltrb.r);      //w
                    encoded_bb[anchor_idx].xcycwh.h = (-p_anchor->ltrb.t + p_anchor->ltrb.b);      //h
                    encoded_labels[anchor_idx] = 0;
                }
            }
        }
    }
}

