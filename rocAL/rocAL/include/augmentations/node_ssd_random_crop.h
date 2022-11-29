#pragma once
#include "node.h"
#include "parameter_factory.h"
#include "parameter_crop_factory.h"


// todo:: move this to common header
template<typename T = std::mt19937, std::size_t state_size = T::state_size>
class SeededRNG {
  /*
  * @param batch_size How many RNGs to store
  * @param state_size How many seed are used to initialize one RNG. Used to lower probablity of
  * collisions between seeds used to initialize RNGs in different operators.
  */
public:
  SeededRNG (int batch_size = 128) {
      std::random_device source;
      _batch_size = batch_size;
      std::size_t _random_data_size = state_size * batch_size ;
      std::vector<std::random_device::result_type> random_data(_random_data_size);
      std::generate(random_data.begin(), random_data.end(), std::ref(source));
      _rngs.reserve(batch_size);
      for (int i=0; i < (int)(_batch_size*state_size); i += state_size) {
        std::seed_seq seeds(std::begin(random_data) + i, std::begin(random_data)+ i +state_size);
        _rngs.emplace_back(T(seeds));
      }
  }

  /**
   * Returns engine corresponding to given sample ID
   */
   T &operator[](int sample) noexcept {
    return _rngs[sample % _batch_size];
  }

private:
    std::vector<T> _rngs;
    int _batch_size;
};

class SSDRandomCropNode : public Node
{
public:
    SSDRandomCropNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    SSDRandomCropNode() = delete;
    void init(FloatParam *crop_area_factor, FloatParam *crop_aspect_ratio, FloatParam *x_drift, FloatParam *y_drift, int num_of_attempts);
    unsigned int get_dst_width() { return _outputs[0]->info().max_shape()[0]; }
    unsigned int get_dst_height() { return _outputs[0]->info().max_shape()[1]; }
    std::shared_ptr<RocalRandomCropParam> get_crop_param() { return _crop_param; }
    float get_threshold(){return _threshold;}
    std::vector<std::pair<float,float>> get_iou_range(){return _iou_range;}
    bool is_entire_iou(){return _entire_iou;}
    void set_meta_data_batch() {}

protected:
    void create_node() override;
    void update_node() override;

private:
    std::shared_ptr<RocalRandomCropParam> _meta_crop_param;
    vx_array _crop_width, _crop_height, _x1, _y1, _x2, _y2;
    std::vector<uint> _crop_width_val, _crop_height_val, _x1_val, _y1_val, _x2_val, _y2_val;
    // unsigned int _dst_width, _dst_height;
    std::vector<RocalROI> in_roi;
    size_t _dest_width;
    size_t _dest_height;
    float  _threshold = 0.05;
    std::vector<std::pair<float,float>> _iou_range;
    int _num_of_attempts = 20;
    bool _entire_iou = false;
    std::shared_ptr<RocalRandomCropParam> _crop_param;
    SeededRNG<std::mt19937, 4> _rngs;     // setting the state_size to 4 for 4 random parameters.
    unsigned _layout, _roi_type;
};