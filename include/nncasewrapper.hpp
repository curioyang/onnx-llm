//
//  ortwrapper.hpp
//
//  Created by zhaode on 2024/10/09.
//  ZhaodeWang
//

#ifndef ORTWRAPPER_hpp
#define ORTWRAPPER_hpp

#include <memory>
#include <nncase/runtime/interpreter.h>
#include <nncase/runtime/runtime_tensor.h>
#include <nncase/runtime/simple_types.h>
#include <nncase/runtime/util.h>
#include <type_traits>
#include <iostream>
#include <fstream>

namespace Ort {

class RuntimeManager {
public:
  RuntimeManager() {}

private:
};

class Module {
public:
  size_t count = 0;
  Module(std::shared_ptr<RuntimeManager> runtime, const std::string &path) {
    std::ifstream ifs(path, std::ios::binary);
    interpreter_.load_model(ifs).unwrap_or_throw();
    entry_function_ = interpreter_.entry_function().unwrap_or_throw();
  }

  nncase::tuple onForward(std::vector<nncase::value_t> &inputs) {
    if (0)
    {
      std::ofstream outputFile("input_desc"+std::to_string(count)+".txt");   
      {
        auto tensor_ = inputs[0].as<nncase::tensor>().expect("not tensor");
        auto data = nncase::runtime::get_output_data(tensor_).unwrap_or_throw();
        auto shape = tensor_->shape();
        auto datasize = 1;
        outputFile<< "fp32: ";
        for(auto ii : shape)
        {
          outputFile<< ii << " ";
          datasize*=ii;
        }
        outputFile<<std::endl;

        std::ofstream oufile("input_ids_float"+std::to_string(count)+".bin", std::ios::binary);
        if (oufile) {
          oufile.write(reinterpret_cast<char*>(data), datasize * sizeof(float));
          oufile.close();
        }
      }

      {
        auto tensor_ = inputs[1].as<nncase::tensor>().expect("not tensor");
        auto data = nncase::runtime::get_output_data(tensor_).unwrap_or_throw();
        auto shape = tensor_->shape();
        auto datasize = 1;
        outputFile<< "fp32: ";
        for(auto ii : shape)
        {
          outputFile<< ii << " ";
          datasize*=ii;
        }
        outputFile<<std::endl;
        std::ofstream oufile("attention_mask_float"+std::to_string(count)+".bin", std::ios::binary);
        if (oufile) {
          oufile.write(reinterpret_cast<char*>(data), datasize * sizeof(float));
          oufile.close();
        }
      }

      {
        auto tensor_ = inputs[2].as<nncase::tensor>().expect("not tensor");
        auto data = nncase::runtime::get_output_data(tensor_).unwrap_or_throw();
        auto shape = tensor_->shape();
        auto datasize = 1;
        outputFile<< "i32: ";
        for(auto ii : shape)
        {
          outputFile<< ii << " ";
          datasize*=ii;
        }
        outputFile<<std::endl;
        std::ofstream oufile("postion_ids_int"+std::to_string(count)+".bin", std::ios::binary);
        if (oufile) {
          oufile.write(reinterpret_cast<char*>(data), datasize * sizeof(int));
          oufile.close();
        }
      }

      {
        auto tensor_ = inputs[3].as<nncase::tensor>().expect("not tensor");
        auto data = nncase::runtime::get_output_data(tensor_).unwrap_or_throw();
        auto shape = tensor_->shape();
        auto datasize = 1;
        outputFile<< "fp32: ";
        for(auto ii : shape)
        {
          outputFile<< ii << " ";
          datasize*=ii;
        }
        outputFile<<std::endl;
        std::ofstream oufile("past_key_values_float"+std::to_string(count)+".bin", std::ios::binary);
        if (oufile) {
          oufile.write(reinterpret_cast<char*>(data), datasize * sizeof(float));
          oufile.close();
        }
      }
      count+=1;
    }
    

    return entry_function_->invoke(inputs)
        .unwrap_or_throw()
        .as<nncase::tuple>()
        .unwrap_or_throw();
  }

private:
  nncase::runtime::interpreter interpreter_;
  nncase::runtime::runtime_function *entry_function_;
};

template <typename T>
static nncase::tensor _Input(const std::vector<int> &shape,
                             std::shared_ptr<RuntimeManager> rtmgr) {
  nncase::dims_t shape_int64(shape.begin(), shape.end());
  return nncase::runtime::hrt::create(
             std::is_same_v<T, float> ? nncase::dt_float32 : nncase::dt_int32,
             shape_int64, nncase::runtime::host_runtime_tensor::pool_shared)
      .unwrap_or_throw()
      .impl();
}

} // namespace Ort

#endif /* ORTWRAPPER_hpp */