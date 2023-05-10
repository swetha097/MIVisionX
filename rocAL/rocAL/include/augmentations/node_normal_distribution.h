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

#pragma once
#include "node.h"
#include "graph.h"
#include <random>

// todo:: move this to common header
template <typename RNG = std::mt19937>
class BatchRNG {
 public:
  /**
   * @brief Used to keep batch of RNGs, so Operators can be immune to order of sample processing
   * while using randomness
   *
   * @param seed Used to generate seed_seq to initialize batch of RNGs
   * @param batch_size How many RNGs to store
   * @param state_size How many seed are used to initialize one RNG. Used to lower probablity of
   * collisions between seeds used to initialize RNGs in different operators.
   */
  BatchRNG(int64_t seed, int batch_size, int state_size = 4) 
  : seed_(seed) {
    std::seed_seq seq{seed_};
    std::vector<uint32_t> seeds(batch_size * state_size);
    seq.generate(seeds.begin(), seeds.end());
    rngs_.reserve(batch_size);
    for (int i = 0; i < batch_size * state_size; i += state_size) {
      std::seed_seq s(seeds.begin() + i, seeds.begin() + i + state_size);
      rngs_.emplace_back(s);
    }
  }


  /**
   * Returns engine corresponding to given sample ID
   */
  RNG &operator[](int sample) noexcept {
    return rngs_[sample];
  }


 private:
  int64_t seed_;
  std::vector<RNG> rngs_;
};



class NormalDistributionNode : public Node
{
public:
    NormalDistributionNode(const std::vector<rocalTensor *> &inputs, const std::vector<rocalTensor *> &outputs);
    NormalDistributionNode() = delete;
    void init(float mean, float stddev);
    void update_param();
protected:
    void create_node() override;
    void update_node() override;
    float _mean, _std_dev;
    std::random_device rd{}; // Check this
    std::mt19937 _generator{12345};
    std::normal_distribution<float> _dist_normal; // Normal Distribution
    std::vector<float> _mean_array, _std_dev_array, _normal_distribution_array;
    unsigned _num_of_dims;
    vx_size * _stride;
    BatchRNG<std::mt19937> _rngs = {89,2};

};
