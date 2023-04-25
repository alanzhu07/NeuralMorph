#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <torch/torch.h>
#include <torch/script.h>
#include "Diffusion.h"
#include <iostream>

//==============================================================================
AudioPluginAudioProcessor::AudioPluginAudioProcessor()
     : AudioProcessor (BusesProperties()
                     #if ! JucePlugin_IsMidiEffect
                      #if ! JucePlugin_IsSynth
                       .withInput  ("Input",  juce::AudioChannelSet::stereo(), true)
                      #endif
                       .withOutput ("Output", juce::AudioChannelSet::stereo(), true)
                     #endif
                       ),
                        thumbnailCache (5),
                        visualizer_sample(512, formatManager, thumbnailCache),
                        visualizer_morph(512, formatManager, thumbnailCache),
                        parameters (*this, nullptr, juce::Identifier ("NeuralMorph"),
                      {
                        std::make_unique<juce::AudioParameterFloat> (juce::ParameterID ("mix", 1),
                                                                       "Mix",
                                                                       0.0f,
                                                                       1.0f,
                                                                       0.5f),
                        std::make_unique<juce::AudioParameterFloat> (juce::ParameterID ("noise", 1),
                                                                       "Noise Amount",
                                                                       0.0f,
                                                                       1.0f,
                                                                       0.5f),
                        std::make_unique<juce::AudioParameterInt> (juce::ParameterID ("steps", 1),
                                                                       "Denoising Steps",
                                                                       5,
                                                                       50,
                                                                       10),
                        std::make_unique<juce::AudioParameterBool> (juce::ParameterID ("morphOn", 1),
                                                                      "Morph On",
                                                                      false)
                      })
{
//    progressPercentage = 0.0;
//    torchThread = TorchThread(&progressPercentage, this);
    torchThread.startThread();
    formatManager.registerBasicFormats();
//    auto morphVoice = new MorphVoice();
//    morphVoice->setBuffer(&fileBuffer);
//    synth.addVoice(&morphVoice);
//    synth.addVoice(dynamic_cast<juce::SynthesiserVoice*>(morphVoice));
    synth.addVoice(new MorphVoice(parameters));
    synth.addSound(new MorphSound());
    
//    visualizer.addChangeListener (this);
//    visualizer.setRepaintRate(1);
//    visualizer.setBufferSize(effective_length);
}


AudioPluginAudioProcessor::~AudioPluginAudioProcessor()
{
//    torchThread.signalThreadShouldExit();
    torchThread.stopThread(10000);
}

//==============================================================================
const juce::String AudioPluginAudioProcessor::getName() const
{
    return JucePlugin_Name;
}

bool AudioPluginAudioProcessor::acceptsMidi() const
{
   #if JucePlugin_WantsMidiInput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::producesMidi() const
{
   #if JucePlugin_ProducesMidiOutput
    return true;
   #else
    return false;
   #endif
}

bool AudioPluginAudioProcessor::isMidiEffect() const
{
   #if JucePlugin_IsMidiEffect
    return true;
   #else
    return false;
   #endif
}

double AudioPluginAudioProcessor::getTailLengthSeconds() const
{
    return 0.0;
}

int AudioPluginAudioProcessor::getNumPrograms()
{
    return 1;   // NB: some hosts don't cope very well if you tell them there are 0 programs,
                // so this should be at least 1, even if you're not really implementing programs.
}

int AudioPluginAudioProcessor::getCurrentProgram()
{
    return 0;
}

void AudioPluginAudioProcessor::setCurrentProgram (int index)
{
    juce::ignoreUnused (index);
}

const juce::String AudioPluginAudioProcessor::getProgramName (int index)
{
    juce::ignoreUnused (index);
    return {};
}

void AudioPluginAudioProcessor::changeProgramName (int index, const juce::String& newName)
{
    juce::ignoreUnused (index, newName);
}

//==============================================================================
void AudioPluginAudioProcessor::prepareToPlay (double sampleRate, int samplesPerBlock)
{
    // Use this method as the place to do any pre-playback
    // initialisation that you need..
    
    synth.setCurrentPlaybackSampleRate (sampleRate);
    midiCollector.reset (sampleRate);
    
    juce::ignoreUnused (sampleRate, samplesPerBlock);
    DBG("sr: " << sampleRate << " perBlock: " << samplesPerBlock);
        
}

bool AudioPluginAudioProcessor::loadSampleClip(juce::File& file) {
        try {
            auto *reader = formatManager.createReaderFor (file);
            if (reader == nullptr) return false;
            
            auto *morphVoice = dynamic_cast<MorphVoice*>(synth.getVoice(0));
            if (morphVoice == nullptr) return false;
            
            morphVoice->fileBuffer.setSize (2, effective_length);
            reader->read(&(morphVoice->fileBuffer), 0, effective_length, 0, true, true);
            torchThread.copyInput (morphVoice->fileBuffer);
            
            morphVoice->inferenceBuffer.setSize (2, effective_length);
            
            visualizer_sample.setSource(&(morphVoice->fileBuffer), 44100.0, 0);
            
            sampleClipLoaded = true;
            sampleName = file.getFileName();
            
            DBG("sample load success");
            delete reader;
            return true;
        }
        catch (const c10::Error& e) {
            return false;
        }
}

bool AudioPluginAudioProcessor::loadTorchModule(juce::File& file) {
    
    torchModuleLoaded = torchThread.loadTorchModule(file);
    if (torchModuleLoaded) modelName = file.getFileNameWithoutExtension();
    return torchModuleLoaded;
}

void AudioPluginAudioProcessor::torchInference() {
    
    jassert(torchModuleLoaded);
    
    torchThread.requestInference();
    outputDrawn = false;
    torchInferenceInProgress = true;
}

void AudioPluginAudioProcessor::torchResample() {
    
    jassert(torchModuleLoaded);
    jassert(sampleClipLoaded);
    
    torchThread.requestInference();
    outputDrawn = false;
    torchInferenceInProgress = true;
    
    
}

void AudioPluginAudioProcessor::setResampleNoiseLevel() {
    auto noiseLevel = (double) *(parameters.getRawParameterValue ("noise"));
    DBG("setting noise to " << noiseLevel);
    torchThread.noise_level = noiseLevel;
}

void AudioPluginAudioProcessor::setInferenceSteps() {
    auto steps = (int) *(parameters.getRawParameterValue ("steps"));
    DBG("setting steps to " << steps);
    torchThread.num_steps = steps;
}


void AudioPluginAudioProcessor::releaseResources()
{
    // When playback stops, you can use this as an opportunity to free up any
    // spare memory, etc.
}

bool AudioPluginAudioProcessor::isBusesLayoutSupported (const BusesLayout& layouts) const
{
  #if JucePlugin_IsMidiEffect
    juce::ignoreUnused (layouts);
    return true;
  #else
    // This is the place where you check if the layout is supported.
    // In this template code we only support mono or stereo.
    // Some plugin hosts, such as certain GarageBand versions, will only
    // load plugins that support stereo bus layouts.
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono()
     && layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // This checks if the input layout matches the output layout
   #if ! JucePlugin_IsSynth
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
   #endif

    return true;
  #endif
}

void AudioPluginAudioProcessor::processBlock (juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages)
{
    
    buffer.clear();
    
    checkInferenceCompleted();
    
    synth.renderNextBlock(buffer, midiMessages, 0, buffer.getNumSamples());
   
}

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    return new AudioPluginAudioProcessorEditor (*this, parameters);
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::ignoreUnused (destData);
//    juce::MemoryOutputStream (destData, true).writeFloat (*gain);
    auto state = parameters.copyState();
    std::unique_ptr<juce::XmlElement> xml (state.createXml());
    copyXmlToBinary (*xml, destData);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ignoreUnused (data, sizeInBytes);
    
    std::unique_ptr<juce::XmlElement> xmlState (getXmlFromBinary (data, sizeInBytes));

    if (xmlState.get() != nullptr)
        if (xmlState->hasTagName (parameters.state.getType()))
            parameters.replaceState (juce::ValueTree::fromXml (*xmlState));
}

bool AudioPluginAudioProcessor::getModuleLoaded () {
    return torchModuleLoaded;
}
bool AudioPluginAudioProcessor::getSampleLoaded () {
    return sampleClipLoaded;
}
bool AudioPluginAudioProcessor::getInferenceCompleted () {
    return torchInferenceCompleted && torchOutputCopied;
}

bool AudioPluginAudioProcessor::getInferenceInProgress () {
    return torchInferenceInProgress;
}

void AudioPluginAudioProcessor::checkInferenceCompleted () {
    
    if (torchThread.inferenceCompleted && !torchThread.outputCopied) {
        auto *morphVoice = dynamic_cast<MorphVoice*>(synth.getVoice(0));
        juce::ScopedLock sl(getCallbackLock());
        juce::MessageManagerLock mml;
        if (mml.lockWasGained()) {
            torchThread.copyOutput (&(morphVoice->inferenceBuffer));
        }
        
    }
    
    if (torchThread.outputCopied && !sampleDrawn) {
        auto *morphVoice = dynamic_cast<MorphVoice*>(synth.getVoice(0));
        juce::ScopedLock sl(getCallbackLock());
        juce::MessageManagerLock mml;
        if (mml.lockWasGained()) {
            visualizer_morph.setSource(&(morphVoice->inferenceBuffer), 44100.0, 1);
            sampleDrawn = true;
        }
    }
    
    torchInferenceCompleted = torchThread.inferenceCompleted;
    torchOutputCopied = torchThread.outputCopied;
    if (torchInferenceInProgress && torchInferenceCompleted) torchInferenceInProgress = false;
    
//    if (torchThread.inferenceCompleted != torchInferenceCompleted) {
//        DBG("check inference");
//        torchInferenceCompleted = torchThread.inferenceCompleted;
//        torchInferenceInProgress = false;
////        buttonListeners.callChecked (checker, [this] (Listener& l) { l.buttonClicked (this); });
////        listeners.call([this] (Listener& l) { l.inferenceStatusChanged (this); });
////        listeners.call(&AudioPluginAudioProcessor::inferenceStatusChanged, this);
//    }
//    if (torchThread.outputCopied != torchOutputCopied ) {
//        DBG("check output copied");
//        torchOutputCopied = torchThread.outputCopied;
////        listeners.call([this] (Listener& l) { l.inferenceStatusChanged (this); });
//        if (torchOutputCopied) {
//            auto *morphVoice = dynamic_cast<MorphVoice*>(synth.getVoice(0));
//            juce::ScopedLock sl(getCallbackLock());
//            juce::MessageManagerLock mml;
//            if (mml.lockWasGained()) {
//                visualizer_morph.setSource(&(morphVoice->inferenceBuffer), 44100.0, 1);
//            }
//        }
//    }
//    if (torchInferenceCompleted && !torchThread.outputCopied) {
//        auto *morphVoice = dynamic_cast<MorphVoice*>(synth.getVoice(0));
//        torchThread.copyOutput (&(morphVoice->inferenceBuffer));
//    }
    
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}
