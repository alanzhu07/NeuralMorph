#pragma once

#include "PluginProcessor.h"
//#include <juce_audio_formats/juce_audio_formats.h>

//==============================================================================
class AudioPluginAudioProcessorEditor  : public juce::AudioProcessorEditor, private juce::ChangeListener
//, private juce::AudioProcessorValueTreeState::Listener
{
public:
    explicit AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor&, juce::AudioProcessorValueTreeState&);
    ~AudioPluginAudioProcessorEditor() override;

    //==============================================================================
    void paint (juce::Graphics&) override;
    void resized() override;
    
    
    juce::Slider mixSlider;
    juce::Slider noiseSlider;
    juce::Slider stepSlider;
    
    juce::Label mixLabel;
    juce::Label noiseLabel;
    juce::Label stepLabel;
    juce::TextButton loadModelButton;
    juce::TextButton loadClipButton;
    juce::TextButton inferenceButton;
    juce::TextButton bypassButton;
    juce::ProgressBar progressBar;
    
    std::unique_ptr<juce::FileChooser> chooser;
    
    
    
    void loadTorchModule();
    void loadSampleClip();
    void checkInferenceOk();
    void doMorph(AudioPluginAudioProcessor& p);
//    void setBypass(AudioPluginAudioProcessor& p);
//    void setMix(AudioPluginAudioProcessor& p);
//    void setNoise();
//    double progressPercentage = 0.0;

private:
    // This reference is provided as a quick way for your editor to
    // access the processor object that created it.
    AudioPluginAudioProcessor& processorRef;
    juce::AudioProcessorValueTreeState& valueTreeState;
    
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> mixAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> noiseAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> stepAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> onAttachment;
//    double *progressPercentage;
    
//    void inferenceStatusChanged(AudioPluginAudioProcessor*) override;
    void changeListenerCallback (juce::ChangeBroadcaster* source) override;
//    void parameterChanged (const juce::String& parameterID, float newValue) override;
    bool sampleLoadingFailed = false;
    bool modelLoadingFailed = false;
    juce::LookAndFeel_V4 myLookAndFeel;
    
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR (AudioPluginAudioProcessorEditor)
};
