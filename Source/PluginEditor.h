#pragma once

#include "PluginProcessor.h"
//#include <juce_audio_formats/juce_audio_formats.h>

//==============================================================================
class AudioPluginAudioProcessorEditor  : public juce::AudioProcessorEditor, public AudioPluginAudioProcessor::Listener
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    
    void inferenceStatusChanged(AudioPluginAudioProcessor*) override;
    
    juce::Slider mixSlider;
    juce::Slider noiseSlider;
    juce::TextButton loadModelButton;
    juce::TextButton loadClipButton;
    juce::TextButton inferenceButton;
    juce::TextButton bypassButton;
    juce::ProgressBar progressBar;
    
    std::unique_ptr<juce::FileChooser> chooser;
    
    
    void loadTorchModule(AudioPluginAudioProcessor& p);
    void loadSampleClip(AudioPluginAudioProcessor& p);
    void checkInferenceOk(AudioPluginAudioProcessor& p);
    void doMorph(AudioPluginAudioProcessor& p);
    void setBypass(AudioPluginAudioProcessor& p);
    void setMix(AudioPluginAudioProcessor& p);
    void setNoise(AudioPluginAudioProcessor& p);
    double progressPercentage = 0.0;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginAudioProcessor& processorRef;
//    double *progressPercentage;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
