#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <torch/torch.h>
#include <iostream>

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p)
: AudioProcessorEditor (&p), progressBar (p.torchThread.progress), processorRef (p)
{
    juce::ignoreUnused (processorRef);
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    
    p.addListener(this);
    
    setSize (600, 600);
    
    mixSlider.setSliderStyle (juce::Slider::RotaryVerticalDrag);
    mixSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, true, 50, 50);
    mixSlider.setRange (0.0, 1.0, 0.01);
    mixSlider.onDragEnd = [this, &p] { setMix(p); };
    mixSlider.setValue (0.5, juce::dontSendNotification);
    addAndMakeVisible (&mixSlider);
    
    noiseSlider.setSliderStyle (juce::Slider::RotaryVerticalDrag);
    noiseSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, true, 50, 50);
    noiseSlider.setRange (0.0, 1.0, 0.01);
    noiseSlider.onDragEnd = [this, &p] { setNoise(p); };
    noiseSlider.setValue (0.5, juce::dontSendNotification);
    addAndMakeVisible (&noiseSlider);
    
    
    loadModelButton.setButtonText ("Select model to load...");
    loadModelButton.onClick = [this, &p] { loadTorchModule(p);  };
    addAndMakeVisible (&loadModelButton);
    
    loadClipButton.setButtonText ("Select sample to load...");
    loadClipButton.onClick = [this, &p] { loadSampleClip(p);  };
    addAndMakeVisible (&loadClipButton);
    
    inferenceButton.setButtonText ("Morph");
    inferenceButton.onClick = [this, &p] { doMorph(p);  };
    inferenceButton.setEnabled (false);
    inferenceButton.setToggleable (true);
    addAndMakeVisible (&inferenceButton);
    
    bypassButton.setButtonText ("OFF");
    bypassButton.setEnabled (false);
    bypassButton.onClick = [this, &p] { setBypass(p);  };
    bypassButton.setClickingTogglesState (true);
    addAndMakeVisible (&bypassButton);
    
    DBG("progressp perc = " << p.torchThread.progress);
//    progressPercentage = &(p.progressPercentage);
//    progressBar = std::make_unique<juce::ProgressBar> (p.torchThread.progress);
//    addAndMakeVisible (progressBar.get());
    addAndMakeVisible (&progressBar);
    
    
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
}

void AudioPluginAudioProcessorEditor::checkInferenceOk(AudioPluginAudioProcessor& p) {
    if (p.getModuleLoaded() && p.getSampleLoaded()) {
        inferenceButton.setEnabled (true);
    }
}

void AudioPluginAudioProcessorEditor::loadTorchModule(AudioPluginAudioProcessor& p)
{
    chooser = std::make_unique<juce::FileChooser> ("Select a saved torch module...",
                                                           juce::File{},
                                                           "*");
    auto chooserFlags = juce::FileBrowserComponent::openMode
                      | juce::FileBrowserComponent::canSelectFiles;

    chooser->launchAsync (chooserFlags, [this, &p] (const juce::FileChooser& fc)
    {
        DBG("choosefile");
        auto file = fc.getResult();
        
        if (file != juce::File{}) {
            DBG("user chose " << file.getFullPathName());
            
            auto success = p.loadTorchModule(file);
            
            if (success)
                loadModelButton.setButtonText ("Model " + file.getFileNameWithoutExtension() + " loaded");
            else loadModelButton.setButtonText ("Model loading failed");
            
            checkInferenceOk(p);
        }
    });
}

void AudioPluginAudioProcessorEditor::loadSampleClip(AudioPluginAudioProcessor& p)
{
    chooser = std::make_unique<juce::FileChooser> ("Select a sample clip...",
                                                           juce::File{},
                                                           "*");
    auto chooserFlags = juce::FileBrowserComponent::openMode
                      | juce::FileBrowserComponent::canSelectFiles;

    chooser->launchAsync (chooserFlags, [this, &p] (const juce::FileChooser& fc)
    {
        DBG("choosefile");
        auto file = fc.getResult();
        
        if (file != juce::File{}) {
            DBG("user chose " << file.getFullPathName());
            
            auto success = p.loadSampleClip(file);
            
            if (success)
                loadClipButton.setButtonText ("Sample " + file.getFileName() + " loaded");
            else loadClipButton.setButtonText ("Sample loading failed");
            
            checkInferenceOk(p);
        }
    });
}

void AudioPluginAudioProcessorEditor::doMorph(AudioPluginAudioProcessor& p) {
//    progressPercentage = 0.0;
//    p.torchResample(&progressPercentage);
    
    p.torchResample();
//    inferenceButton.setToggleState(true, juce::dontSendNotification);
//    progressPercentage = 0.5;
}

void AudioPluginAudioProcessorEditor::setMix(AudioPluginAudioProcessor& p) {
//    progressPercentage = 0.0;
//    p.torchResample(&progressPercentage);
    auto mixParam = mixSlider.getValue();
    DBG("slider val " << mixParam);
    p.setMix((float) mixParam);
//    inferenceButton.setToggleState(true, juce::dontSendNotification);
//    progressPercentage = 0.5;
}

void AudioPluginAudioProcessorEditor::setNoise(AudioPluginAudioProcessor& p) {
    
    auto noiseParam = noiseSlider.getValue();
    DBG("noiseval " << noiseParam);
    p.setResampleNoiseLevel(noiseParam);
    
}

void AudioPluginAudioProcessorEditor::setBypass(AudioPluginAudioProcessor& p) {
    auto state = bypassButton.getToggleState();
    juce::String stateString  = state ? "ON" : "OFF";
    DBG("bypass " << stateString);
    
    if (state) {
        bypassButton.setButtonText ("ON");
        p.setMorph(true);
    } else {
        bypassButton.setButtonText ("OFF");
        p.setMorph(false);
    }
}

void AudioPluginAudioProcessorEditor::inferenceStatusChanged(AudioPluginAudioProcessor* p) {
    
    DBG("callback " << (int) (p->getInferenceInProgress()));
    if (p->getInferenceInProgress()) {
        inferenceButton.setEnabled (false);
        bypassButton.setEnabled (false) ;
    }
    else if (p->getModuleLoaded() && p->getSampleLoaded() && p->getInferenceCompleted()) {
        juce::ScopedLock sl(p->getCallbackLock());
        juce::MessageManagerLock mml;
        if (mml.lockWasGained()) {
            inferenceButton.setEnabled (true);
//            inferenceButton.setToggleState(false, juce::dontSendNotification);
            bypassButton.setEnabled (true) ;
            
        }
    }
    else {
        juce::ScopedLock sl(p->getCallbackLock());
        juce::MessageManagerLock mml;
        if (mml.lockWasGained()) {
            inferenceButton.setEnabled (false);
            bypassButton.setEnabled (false) ;
        }
    }
    
}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    
//    g.setColour (juce::Colours::white);
//    g.setFont (15.0f);
//    g.drawFittedText ("Hello World!", getLocalBounds(), juce::Justification::centred, 1);
    
    
}

void AudioPluginAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    mixSlider.setBounds (40, 30, 200, 200);
    noiseSlider.setBounds (100, 50, 200, 200);
    loadModelButton.setBounds (40, 300, getWidth() - 20, 20);
    loadClipButton.setBounds (40, 320, getWidth() - 20, 20);
    inferenceButton.setBounds (10, 400, getWidth() - 20, 20);
    bypassButton.setBounds( 350, 350, 75, 75);
    progressBar.setBounds(25, 100, getWidth() - 25, 25);
    
}
