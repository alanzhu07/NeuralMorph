#pragma once

//#include <juce_audio_processors/juce_audio_processors.h>
#include <JuceHeader.h>
#include <torch/torch.h>
#include "TorchThread.cpp"

//#define JUCE_MODAL_LOOPS_PERMITTED = 1

//==============================================================================
struct MorphSound   : public juce::SynthesiserSound
{
    MorphSound() {
        DBG("sound added");
    }

    bool appliesToNote    (int) override        { return true; }
    bool appliesToChannel (int) override        { return true; }
};

//==============================================================================
class MorphVoice : public juce::SynthesiserVoice
{
public:
    MorphVoice() {}
//    ~MorphVoice() {} override;
    
    
    bool canPlaySound(juce::SynthesiserSound* sound) override
    {
        DBG("can play sound?");
        return dynamic_cast<MorphSound*>(sound) != nullptr;
    }
    
    void startNote(int midiNoteNumber, float velocity, juce::SynthesiserSound* sound, int currentPitchWheelPosition) override
    {
        DBG("received start note");
        currentAngle = 0.0;
        level = velocity * 0.15;
        tailOff = 0.0;
        currSample = 0;
        playing = true;
    }
    
    void stopNote (float /*velocity*/, bool allowTailOff) override
    {
        DBG("received stop note");
        playing = false;
        if (allowTailOff)
        {
            if (tailOff == 0.0)
                tailOff = 1.0;
        }
        else
        {
            clearCurrentNote();
            angleDelta = 0.0;
        }
    }
    
    void pitchWheelMoved(int /*newValue*/) override {}
    void controllerMoved(int /*controllerNumber*/, int /*newValue*/) override {}
    
    void renderNextBlock(juce::AudioBuffer<float>& outputBuffer, int startSample, int numSamples) override
    {
//        if (fileBuffer != nullptr)
        if (addMorphed) {
            DBG("morphed");
        }
        if (playing)
        {
            auto outputSamplesRemaining = numSamples;
            auto outputSamplesOffset = startSample;
            auto numOutputChannels = outputBuffer.getNumChannels();
            
//            DBG("rendering within morphvoice");
            
            while ((currSample < fileBuffer.getNumSamples()) && (outputSamplesRemaining > 0)) {
                
                DBG("position: " << currSample << " remaning: " << outputSamplesRemaining <<
                    " file buffer size: " << fileBuffer.getNumSamples());
                
                auto bufferSamplesRemaining = fileBuffer.getNumSamples() - currSample;
                auto samplesThisTime = juce::jmin (numSamples, bufferSamplesRemaining);
                
                for (int channel = 0; channel < numOutputChannels; ++channel)
                {
                    
                    
                    
                    if (addMorphed) {
//                        outputBuffer.copyFrom(channel, outputSamplesOffset, inferenceBuffer, channel, currSample, samplesThisTime);
//                        outputBuffer.applyGain (channel, outputSamplesOffset, samplesThisTime, mixParam) ;
                        outputBuffer.addFrom(channel, outputSamplesOffset, inferenceBuffer, channel, currSample, samplesThisTime, mixParam);
                        outputBuffer.addFrom(channel, outputSamplesOffset, fileBuffer, channel, currSample, samplesThisTime, 1.0f - mixParam);
                    }
                    else {
                        outputBuffer.copyFrom(channel, outputSamplesOffset, fileBuffer, channel, currSample, samplesThisTime);
                    }
                    
                    //                if (torchInferenceCompleted && torchThread.outputCopied) {
                    //                    buffer.copyFrom(channel, 0, inferenceBuffer, channel, curr_sample, samplesThisTime);
                    //                } else if (sampleClipLoaded) {
                    //                    buffer.copyFrom(channel, 0, fileBuffer, channel, curr_sample, samplesThisTime);
                    //                }
                }
                
                outputSamplesRemaining -= samplesThisTime;
                outputSamplesOffset += samplesThisTime;
                currSample += samplesThisTime;
                if (currSample >= fileBuffer.getNumSamples()) {
                    break;
                }
            }
        }
    }
    
    juce::AudioSampleBuffer fileBuffer;
    juce::AudioSampleBuffer inferenceBuffer;
    bool addMorphed = false;
    float mixParam = 1.0f;
    
private:
//    std::shared_ptr<juce::AudioBuffer<float>> sampleBuffer;
    
    bool playing = false;
    int currSample = 0;
    double currentAngle = 0.0, angleDelta = 0.0, level = 0.0, tailOff = 0.0;
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (MorphVoice)
};

//==============================================================================
class AudioPluginAudioProcessor  : public juce::AudioProcessor
{
public:
    //==============================================================================
    AudioPluginAudioProcessor();
    ~AudioPluginAudioProcessor() override;

    //==============================================================================
    void prepareToPlay (double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported (const BusesLayout& layouts) const override;

    void processBlock (juce::AudioBuffer<float>&, juce::MidiBuffer&) override;
    using AudioProcessor::processBlock;

    //==============================================================================
    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    //==============================================================================
    const juce::String getName() const override;

    bool acceptsMidi() const override;
    bool producesMidi() const override;
    bool isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    //==============================================================================
    int getNumPrograms() override;
    int getCurrentProgram() override;
    void setCurrentProgram (int index) override;
    const juce::String getProgramName (int index) override;
    void changeProgramName (int index, const juce::String& newName) override;

    //==============================================================================
    void getStateInformation (juce::MemoryBlock& destData) override;
    void setStateInformation (const void* data, int sizeInBytes) override;

    
    bool loadTorchModule (juce::File& file);
    bool loadSampleClip (juce::File& file);
    
    bool getModuleLoaded ();
    bool getSampleLoaded ();
    bool getInferenceCompleted ();
    bool getInferenceInProgress ();
    
    void torchInference () ;
    void torchResample () ;
    void setMorph (bool morph) ;
    void setMix (float mix);
    void setResampleNoiseLevel (double noiseLevel) ;
    
    void checkInferenceCompleted ();
    
    double progressPercentage = 0.0;
    
    TorchThread torchThread {};
    
    class Listener
       {
       public:
           virtual ~Listener() = default;
           virtual void inferenceStatusChanged(AudioPluginAudioProcessor*) = 0;
       };
    
    void addListener (Listener* newListener);
    
    
    
private:
    
//    MorphVoice *morphVoice;
    
    juce::ListenerList<Listener> listeners;
    
    juce::MidiMessageCollector midiCollector;
    
    juce::AudioFormatManager formatManager;
//    juce::AudioSampleBuffer fileBuffer;
//    juce::AudioSampleBuffer inferenceBuffer;
    
    juce::Synthesiser synth;
    
    torch::Tensor tensor_buffer;
    torch::jit::Module torchModule;
    
    bool torchModuleLoaded = false;
    bool torchInferenceCompleted = false;
    bool torchInferenceInProgress = false;
    bool torchOutputCopied = false;
    bool sampleClipLoaded = false;
    
    int curr_sample = 0;
    int sample_size = 65536;
    int batch_size = 1;
    int effective_length = 1 * sample_size;
//    int num_steps = 25;
    int num_steps = 5;
    
    
    
    //==============================================================================
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessor)
};
