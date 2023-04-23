#include <torch/torch.h>
#include <torch/script.h>
#include <JuceHeader.h>

torch::Tensor sample(torch::jit::script::Module module,
                    torch::Tensor noise,
                    int num_steps,
                    double *progress,
                    juce::Thread *thread);

torch::Tensor resample(torch::jit::script::Module module,
                    torch::Tensor noise,
                    torch::Tensor audio,
                    int num_steps,
                    double noise_level,
                    double *progress,
                    juce::Thread *thread);

void t_to_alpha_sigma(torch::Tensor t, 
                    torch::Tensor* alpha_p,
                    torch::Tensor* sigma_p);

torch::Tensor get_crash_schedule(int steps);
