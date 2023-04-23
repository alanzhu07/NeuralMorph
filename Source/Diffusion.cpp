#include <torch/torch.h>
#include <torch/script.h>
#include "Diffusion.h"

torch::Tensor sample(torch::jit::script::Module module,
                    torch::Tensor noise,
                    int num_steps,
                    double *progress,
                    juce::Thread *thread)
{
    torch::Tensor alphas, sigmas, v, pred, eps;
    
    auto t = noise.new_ones({torch::size(noise, 0)});
    auto steps = get_crash_schedule(num_steps);


    t_to_alpha_sigma(steps, &alphas, &sigmas);
    
    for (int i = 0; i < num_steps; i++) {
        
        if (thread->threadShouldExit()) {
            DBG("exiting");
            return pred;
        }
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(noise);
        inputs.push_back(t * steps[i]);
        v = module.forward(inputs).toTensor();

        // no eta
        pred = noise * alphas[i] - v * sigmas[i];
        if (i < num_steps - 1) {
            eps = noise * sigmas[i] + v * alphas[i];
            noise = pred * alphas[i + 1] + eps * sigmas[i + 1];
        }
        
        *progress = (i + 1) / (double) num_steps;
        
    }

    return pred.clamp(-1, 1);
}

torch::Tensor resample(torch::jit::script::Module module,
                    torch::Tensor noise,
                    torch::Tensor audio,
                    int num_steps,
                    double noise_level,
                    double *progress,
                    juce::Thread *thread)
{
    torch::Tensor alphas, sigmas, v, pred, eps;
    
    auto t = noise.new_ones({torch::size(noise, 0)});
    auto steps = get_crash_schedule(num_steps);
    steps = steps.masked_select(steps < noise_level);
    num_steps = (int)torch::size(steps, 0);
    DBG("num_steps " << num_steps);
    
    if (num_steps == 0) {
        return audio;
    }

    t_to_alpha_sigma(steps, &alphas, &sigmas);
    
    noise = audio * alphas[0] + noise * sigmas[0];
    
    for (int i = 0; i < num_steps; i++) {
        
        if (thread->threadShouldExit()) {
            DBG("exiting");
            return pred;
        }
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(noise);
        inputs.push_back(t * steps[i]);
        v = module.forward(inputs).toTensor();

        // no eta
        pred = noise * alphas[i] - v * sigmas[i];
        if (i < num_steps - 1) {
            eps = noise * sigmas[i] + v * alphas[i];
            noise = pred * alphas[i + 1] + eps * sigmas[i + 1];
        }
        
        *progress =  (i + 1) / (double) num_steps;
    }

    return pred.clamp(-1, 1);
}

void t_to_alpha_sigma(torch::Tensor t, 
                    torch::Tensor* alphas_p,
                    torch::Tensor* sigmas_p)
{
    *alphas_p = torch::cos(t * M_PI / 2);
    *sigmas_p = torch::sin(t * M_PI / 2);
}

torch::Tensor get_crash_schedule(int steps) {
    auto t = torch::linspace(1, 0, steps + 1).slice(0, 0, steps);
    auto sigma = torch::sin(t * M_PI / 2).square();
    auto alpha = (1 - sigma.square()).sqrt();
    t = torch::atan2(sigma, alpha) / M_PI * 2;
    return t;
}
