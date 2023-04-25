#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <torch/torch.h>
#include <iostream>

//==============================================================================
AudioPluginAudioProcessorEditor::AudioPluginAudioProcessorEditor (AudioPluginAudioProcessor& p, juce::AudioProcessorValueTreeState& vts)
: AudioProcessorEditor (&p), progressBar (p.torchThread.progress), processorRef (p),
valueTreeState (vts)
{
    juce::ignoreUnused (processorRef);
    // Make sure that before the constructor has finished, you've set the
    // editor's size to whatever you need it to be.
    
//    p.addListener(this);
//    myLookAndFeel.setColourScheme( juce::LookAndFeel_V4::getMidnightColourScheme ());
//    for (auto font : juce::Font::findAllTypefaceNames())
//        DBG(font);
    
    
    myLookAndFeel.setColourScheme( juce::LookAndFeel_V4::getMidnightColourScheme ());
    setLookAndFeel (&myLookAndFeel);
    mixSlider.setColour( juce::Slider::textBoxOutlineColourId  , juce::Colours::transparentWhite);
    noiseSlider.setColour( juce::Slider::textBoxOutlineColourId  , juce::Colours::transparentWhite);
    stepSlider.setColour( juce::Slider::textBoxOutlineColourId  , juce::Colours::transparentWhite);
//    mixSlider.setColour( juce::Slider::textBoxTextColourId , juce::Colours::red);
    getLookAndFeel().setColour( juce::Slider::thumbColourId , juce::Colour (0xff7f5cbc));
//    setLookAndFeel (&myLookAndFeel);
    
    setSize (1000, 600);
    
    mixSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    mixSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 50, 20);
    mixSlider.setRange (0.0, 1.0, 0.01);
//    mixSlider.onValueChange = [this, &p] { setMix(p); };
//    mixSlider.setValue (0.5, juce::dontSendNotification);
    mixAttachment.reset (new juce::AudioProcessorValueTreeState::SliderAttachment (valueTreeState, "mix", mixSlider));
    addAndMakeVisible (&mixSlider);
    mixLabel.setText ("Mix", juce::dontSendNotification);
    mixLabel.setJustificationType (juce::Justification::centred );
    mixLabel.attachToComponent (&mixSlider, false);
    addAndMakeVisible (&mixLabel);
    
    noiseSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    noiseSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 50, 20);
    noiseSlider.setRange (0.0, 1.0, 0.01);
    noiseSlider.onValueChange = [this] { processorRef.setResampleNoiseLevel(); };
//    noiseSlider.setValue (0.5, juce::dontSendNotification);
    noiseAttachment.reset (new juce::AudioProcessorValueTreeState::SliderAttachment (valueTreeState, "noise", noiseSlider));
    addAndMakeVisible (&noiseSlider);
    noiseLabel.setText ("Noise Amount", juce::dontSendNotification);
    noiseLabel.setJustificationType (juce::Justification::centred );
    noiseLabel.attachToComponent (&noiseSlider, false);
    addAndMakeVisible (&noiseLabel);
    
    stepSlider.setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
    stepSlider.setTextBoxStyle (juce::Slider::TextBoxBelow, false, 50, 20);
    stepSlider.setRange (5, 50, 1);
    stepSlider.setSkewFactorFromMidPoint (10);
    stepSlider.onValueChange = [this] { processorRef.setInferenceSteps(); };
    stepAttachment.reset (new juce::AudioProcessorValueTreeState::SliderAttachment (valueTreeState, "steps", stepSlider));
    addAndMakeVisible (&stepSlider);
    stepLabel.setText ("Denoising Steps", juce::dontSendNotification);
    stepLabel.setJustificationType (juce::Justification::centred );
    stepLabel.attachToComponent (&stepSlider, false);
    addAndMakeVisible (&stepLabel);
    
    loadModelButton.onClick = [this] { loadTorchModule();  };
    addAndMakeVisible (&loadModelButton);
    
    loadClipButton.onClick = [this] { loadSampleClip();  };
    addAndMakeVisible (&loadClipButton);
    
    inferenceButton.setButtonText ("Morph");
    inferenceButton.onClick = [this, &p] { doMorph(p);  };
//    inferenceButton.setEnabled (false);
    inferenceButton.setToggleable (true);
    addAndMakeVisible (&inferenceButton);
    
    
    
    bypassButton.setClickingTogglesState (true);
    onAttachment.reset (new juce::AudioProcessorValueTreeState::ButtonAttachment (valueTreeState, "morphOn", bypassButton));
    addAndMakeVisible (&bypassButton);
    
//    DBG("progressp perc = " << p.torchThread.progress);
//    progressPercentage = &(p.progressPercentage);
//    progressBar = std::make_unique<juce::ProgressBar> (p.torchThread.progress);
//    addAndMakeVisible (progressBar.get());
    addAndMakeVisible (&progressBar);
    
//    addAndMakeVisible (processorRef.visualizer);
    processorRef.visualizer_sample.addChangeListener (this);
    processorRef.visualizer_morph.addChangeListener (this);
//    processorRef.parameters.addParameterListener ("morphOn", this);
    
}

AudioPluginAudioProcessorEditor::~AudioPluginAudioProcessorEditor()
{
    setLookAndFeel (nullptr);
}

void AudioPluginAudioProcessorEditor::changeListenerCallback (juce::ChangeBroadcaster* source) {
    if (source == &(processorRef.visualizer_sample))  {
        DBG("received visualizer sample change");
        repaint();
    } else if (source == &(processorRef.visualizer_morph))  {
        DBG("received visualizer morph change");
        repaint();
    }
}

void AudioPluginAudioProcessorEditor::checkInferenceOk() {
    
    if (processorRef.getModuleLoaded() && processorRef.getSampleLoaded()) {
        if (processorRef.getInferenceInProgress()) {
            inferenceButton.setEnabled (false);
            bypassButton.setEnabled (true) ;
        }
        else if (processorRef.getInferenceCompleted()) {
            inferenceButton.setEnabled (true);
            bypassButton.setEnabled (true);
        }
        else {
            inferenceButton.setEnabled (true);
            bypassButton.setEnabled (false) ;
        }
    } else {
        inferenceButton.setEnabled (false);
        bypassButton.setEnabled (false) ;
    }
}

void AudioPluginAudioProcessorEditor::loadTorchModule()
{
    chooser = std::make_unique<juce::FileChooser> ("Select a saved torch module...",
                                                           juce::File{},
                                                           "*.pt");
    auto chooserFlags = juce::FileBrowserComponent::openMode
                      | juce::FileBrowserComponent::canSelectFiles;

    chooser->launchAsync (chooserFlags, [this] (const juce::FileChooser& fc)
    {
        DBG("choosefile");
        auto file = fc.getResult();
        
        if (file != juce::File{}) {
            DBG("user chose " << file.getFullPathName());
            
            modelLoadingFailed = !processorRef.loadTorchModule(file);
        }
    });
}

void AudioPluginAudioProcessorEditor::loadSampleClip()
{
    chooser = std::make_unique<juce::FileChooser> ("Select a sample clip...",
                                                           juce::File{},
                                                           "*");
    auto chooserFlags = juce::FileBrowserComponent::openMode
                      | juce::FileBrowserComponent::canSelectFiles;

    chooser->launchAsync (chooserFlags, [this] (const juce::FileChooser& fc)
    {
        DBG("choosefile");
        auto file = fc.getResult();
        
        if (file != juce::File{}) {
            DBG("user chose " << file.getFullPathName());
            
            sampleLoadingFailed = !processorRef.loadSampleClip(file);
        }
    });
}

void AudioPluginAudioProcessorEditor::doMorph(AudioPluginAudioProcessor& p) {
    p.torchResample();
}

//void AudioPluginAudioProcessorEditor::setMix(AudioPluginAudioProcessor& p) {
////    progressPercentage = 0.0;
////    p.torchResample(&progressPercentage);
//    auto mixParam = mixSlider.getValue();
//    DBG("slider val " << mixParam);
//    p.setMix((float) mixParam);
////    inferenceButton.setToggleState(true, juce::dontSendNotification);
////    progressPercentage = 0.5;
//}

//void AudioPluginAudioProcessorEditor::setNoise() {
//    
////    auto noiseParam = noiseSlider.getValue();
////    DBG("noiseval " << noiseParam);
////    p.setResampleNoiseLevel(noiseParam);
//    
//    processorRef.setResampleNoiseLevel();
//    
//}

//void AudioPluginAudioProcessorEditor::Steps() {
//    
//    processorRef.setInferenceSteps();
//    
//}

//void AudioPluginAudioProcessorEditor::setBypass(AudioPluginAudioProcessor& p) {
//    auto state = bypassButton.getToggleState();
//    juce::String stateString  = state ? "ON" : "OFF";
//    DBG("bypass " << stateString);
//
//    if (state) {
//        bypassButton.setButtonText ("ON");
//        p.setMorph(true);
//    } else {
//        bypassButton.setButtonText ("OFF");
//        p.setMorph(false);
//    }
//}

//void AudioPluginAudioProcessorEditor::inferenceStatusChanged(AudioPluginAudioProcessor* p) {
//    
//}

//void AudioPluginAudioProcessorEditor::parameterChanged (const juce::String& parameterID, float newValue) {
//    if (parameterID == "morphOn") {
//        DBG(parameterID << newValue);
//        if (bool(newValue)) {
//            bypassButton.setButtonText ("ON");
//        } else bypassButton.setButtonText ("OFF");
//    }
//}

//==============================================================================
void AudioPluginAudioProcessorEditor::paint (juce::Graphics& g)
{
    // (Our component is opaque, so we must completely fill the background with a solid colour)
    g.fillAll (getLookAndFeel().findColour (juce::ResizableWindow::backgroundColourId));
    
    juce::Rectangle<int> thumbnailBounds (getWidth() * 0.05, getHeight() * 0.28, getWidth() * 0.70, getHeight() * 0.3);
    juce::Rectangle<int> thumbnailBounds2 (getWidth() * 0.05, getHeight() * 0.67, getWidth() * 0.70, getHeight() * 0.3);
    g.setColour (getLookAndFeel().findColour ( juce::TextButton::buttonOnColourId));
    g.setOpacity (0.3);
    g.fillRect (thumbnailBounds);
    g.fillRect (thumbnailBounds2);
    
    if (processorRef.visualizer_sample.getNumChannels() != 0) {
        g.setColour (juce::Colour (0xff7f5cbc));
        processorRef.visualizer_sample.drawChannels (g, thumbnailBounds, 0.0, processorRef.visualizer_sample.getTotalLength(), 1.0f);
    }
    if (processorRef.visualizer_morph.getNumChannels() != 0) {
        g.setColour (juce::Colour (0xff7f5cbc));
        processorRef.visualizer_morph.drawChannels (g, thumbnailBounds2, 0.0, processorRef.visualizer_morph.getTotalLength(), 1.0f);
    }
//    g.setColour (juce::Colours::white);
    g.setFont (35.0f);
    g.setColour (juce::Colour (0xe67f5cbc));
    g.drawFittedText ("Neural Morph", getLocalBounds().withHeight(getHeight() * 0.2).withCentre(juce::Point<int>(getWidth()*0.5, (int)getHeight()*0.1)), juce::Justification::centred, 1);
    
    if (processorRef.getSampleLoaded()) loadClipButton.setButtonText ("Sample " + processorRef.sampleName + " loaded");
    else if (sampleLoadingFailed) loadClipButton.setButtonText ("Sample loading failed");
    else loadClipButton.setButtonText ("Select sample to load...");
    
    if (processorRef.getModuleLoaded()) loadModelButton.setButtonText ("Model " + processorRef.modelName + " loaded");
    else if (modelLoadingFailed) loadModelButton.setButtonText ("Model loading failed");
    else loadModelButton.setButtonText ("Select model to load...");
    
    bool addMorphed = (bool) *(valueTreeState.getRawParameterValue ("morphOn"));
    bypassButton.setButtonText (addMorphed ? "ON" : "OFF");
    
    checkInferenceOk();
}

void AudioPluginAudioProcessorEditor::resized()
{
    // This is generally where you'll want to lay out the positions of any
    // subcomponents in your editor..
    mixSlider.setBounds (getWidth() * 0.77, getHeight() * 0.25, getWidth() * 0.25, getHeight() * 0.2);
    noiseSlider.setBounds (getWidth() * 0.77, getHeight() * 0.51, getWidth() * 0.25, getHeight() * 0.2);
    stepSlider.setBounds (getWidth() * 0.77, getHeight() * 0.77, getWidth() * 0.25, getHeight() * 0.2);
    
    loadClipButton.setBounds (getWidth() * 0.05, getHeight() * 0.22, getWidth() * 0.34, getHeight() * 0.05);
    loadModelButton.setBounds (getWidth() * 0.41, getHeight() * 0.22, getWidth() * 0.34, getHeight() * 0.05);
    
    inferenceButton.setBounds (getWidth() * 0.05, getHeight() * 0.02, getHeight() * 0.18, getHeight() * 0.18);
    bypassButton.setBounds (getWidth() * 0.84, getHeight() * 0.02, getHeight() * 0.18, getHeight() * 0.18);
//    getWidth() * 0.05, getHeight() * 0.65, getWidth() * 0.70, getHeight() * 0.3);
    progressBar.setBounds(getWidth() * 0.05, getHeight() * 0.60, getWidth() * 0.70, getHeight() * 0.05);
//    processorRef.visualizer.setBounds(getLocalBounds().withSizeKeepingCentre(getWidth() * 0.75, getHeight() * 0.5));
}
