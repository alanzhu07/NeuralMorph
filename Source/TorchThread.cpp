//
//  TorchThread.cpp
//  NeuralMorph
//
//  Created by Alan Zhu on 2023/4/21.
//

//#include "PluginProcessor.h"
#include <JuceHeader.h>
#include <torch/torch.h>
#include <torch/script.h>
#include "Diffusion.h"

class TorchThread : public juce::Thread {
public:
    TorchThread() : juce::Thread ("torch thread") {
        input_tensor = torch::zeros({1, 2, effective_length}, torch::kFloat32);
//        output_tensor = torch::zeros({1, 2, effective_length}, torch::kFloat32);
    }
    
    bool loadTorchModule(juce::File& file) {
            try {
                torchModule = torch::jit::load(file.getFullPathName().toStdString());
                torchModuleLoaded = true;
                return true;
            }
            catch (const c10::Error& e) {
                torchModuleLoaded = false;
                return false;
            }
    }
    
    void copyInput (juce::AudioSampleBuffer inBuffer) {
        DBG("torch thread copyInput begin");
        
//        input_tensor = torch::zeros({1, 2, effective_length}, torch::kFloat32);
        inferenceCompleted = false;
        
        for (int channel = 0; channel < 2; ++channel) {
            auto* channelData = inBuffer.getReadPointer (channel);
            for (int sample = 0; sample < effective_length; ++sample) {
                input_tensor[0][channel][sample] = channelData[sample];
            }
        }
        
        sampleClipLoaded = true;
        DBG("torch thread copyInput success");
    }
    
    void copyOutput (juce::AudioSampleBuffer *outBuffer) {
        DBG("copy output begin");
        
        jassert(inferenceCompleted);
        
        for (int channel = 0; channel < 2; ++channel) {
            auto* channelData = outBuffer->getWritePointer (channel);
            for (int sample = 0; sample < effective_length; ++sample) {
                channelData[sample] = output_tensor[0][channel][sample].item<float>();
//                DBG("channel " << channel << " sample " << sample << " val " << channelData[sample]);
            }
        }
        
        outputCopied = true;
        DBG("copy output success");
    }
    
    void torchResample() {
        DBG("torch thread resample begin");
        
        progress = 0.0;
        inferenceCompleted = false;
        inferenceRequested = false;
        outputCopied = false;
        
//        auto noise = torch::randn_like(input_tensor);
        auto noise = torch::randn({batch_size, 2, effective_length}, torch::kFloat32);
        
        output_tensor = resample(torchModule, noise, input_tensor, num_steps, noise_level, &progress, this);
        
        
        progress = 1.0;
        
        inferenceCompleted = true;
        
        DBG("torch thread resample success");
    }
    
    void torchInfer() {
        DBG("torch thread infer begin");
        
        progress = 0.0;
        inferenceCompleted = false;
        inferenceRequested = false;
        outputCopied = false;
        
        auto noise = torch::randn({batch_size, 2, effective_length}, torch::kFloat32);
        
        output_tensor = sample(torchModule, noise, num_steps, &progress, this);
        
        progress = 1.0;
        
        inferenceCompleted = true;
        
        DBG("torch thread infer success");
    }
    
    
    
    void requestInference() {
        inferenceRequested = true;
    }
    
    void run () override {
        while (!threadShouldExit()) {
            if (torchModuleLoaded && sampleClipLoaded && inferenceRequested) {
                torchResample();
//                torchInfer();
            }
            else wait(5);
        }
    }
    
    torch::Tensor input_tensor, output_tensor;
    torch::jit::Module torchModule;
    int num_steps = 10;
    double noise_level = 0.5;
    double progress = 0.0;
    bool torchModuleLoaded = false;
    bool inferenceRequested = false;
    bool inferenceCompleted = false;
    bool sampleClipLoaded = false;
    bool outputCopied = false;
    
    int curr_sample = 0;
    int sample_size = 65536;
    int batch_size = 1;
    int effective_length = 1 * sample_size;
};
