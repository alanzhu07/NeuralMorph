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
                       )
{
//    progressPercentage = 0.0;
//    torchThread = TorchThread(&progressPercentage, this);
    torchThread.startThread();
    formatManager.registerBasicFormats();
//    auto morphVoice = new MorphVoice();
//    morphVoice->setBuffer(&fileBuffer);
//    synth.addVoice(&morphVoice);
//    synth.addVoice(dynamic_cast<juce::SynthesiserVoice*>(morphVoice));
    synth.addVoice(new MorphVoice());
    synth.addSound(new MorphSound());
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
            
            morphVoice->fileBuffer.setSize (2, effective_length);
            reader->read(&(morphVoice->fileBuffer), 0, effective_length, 0, true, true);
            
//            morphVoice.fileBuffer.setSize ((int) reader->numChannels, (int) reader->lengthInSamples);
//            reader->read(&(morphVoice.fileBuffer), 0, (int) reader->lengthInSamples, 0, true, true);
            morphVoice->inferenceBuffer.setSize (2, effective_length);
            torchThread.copyInput (morphVoice->fileBuffer);
            
//            fileBuffer.setSize ((int) reader->numChannels, (int) reader->lengthInSamples);
//            reader->read(&fileBuffer, 0, (int) reader->lengthInSamples, 0, true, true);
//            inferenceBuffer.setSize (2, effective_length);
            
//            torchThread.copyInput (fileBuffer);
            
            sampleClipLoaded = true;
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
    return torchModuleLoaded;
}

void AudioPluginAudioProcessor::torchInference() {
    
    jassert(torchModuleLoaded);
    
    torchThread.requestInference();
    torchInferenceInProgress = true;
    listeners.call([this] (Listener& l) { l.inferenceStatusChanged (this); });
}

void AudioPluginAudioProcessor::torchResample() {
    
    jassert(torchModuleLoaded);
    jassert(sampleClipLoaded);
    
    torchThread.requestInference();
    torchInferenceInProgress = true;
    listeners.call([this] (Listener& l) { l.inferenceStatusChanged (this); });
    
    
}

void AudioPluginAudioProcessor::setResampleNoiseLevel(double noiseLevel) {
    
    jassert(noiseLevel >= 0 && noiseLevel <= 1);
    torchThread.noise_level = noiseLevel;
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
//    juce::ignoreUnused (midiMessages);
    
    buffer.clear();

    // Send MIDI messages to the synthesizer
//    juce::MidiBuffer processedMidi;

//    for (const auto metadata : midiMessages)
//    {
//        auto message = metadata.getMessage();
//        const auto time = metadata.samplePosition;
//        DBG("Note number " << message.getNoteNumber());
//
//        if (message.isNoteOn())
//        {
//            DBG("note on!");
////            message = juce::MidiMessage::noteOn(message.getChannel(), message.getNoteNumber(), 1.0f);
//        }
//
////        processedMidi.addEvent(message, time);
//    }

//    midiMessages.swapWith(processedMidi);
    
    checkInferenceCompleted();
//    if (torchInferenceCompleted && !torchThread.outputCopied) {
//            torchThread.copyOutput (&(morphVoice.inferenceBuffer));
////            torchInferenceCompleted = true;
//    }
//    if (torchThread.outputCopied) {
////        DBG("adding morphed");
//        morphVoice.addMorphed = true;
//    }
    
    
    synth.renderNextBlock(buffer, midiMessages, 0, buffer.getNumSamples());
    
//    juce::ScopedNoDenormals noDenormals;
//    auto totalNumOutputChannels = getTotalNumOutputChannels();
//
//    if (torchThread.inferenceCompleted && !torchThread.outputCopied) {
//        torchThread.copyOutput (&inferenceBuffer);
//        torchInferenceCompleted = true;
//    }
//
//    auto bufferSamplesRemaining = fileBuffer.getNumSamples() - curr_sample;
//    auto samplesThisTime = juce::jmin (buffer.getNumSamples(), bufferSamplesRemaining);
//
//    for (int channel = 0; channel < totalNumOutputChannels; ++channel)
//    {
//        auto* channelData = buffer.getWritePointer (channel);
//        juce::ignoreUnused (channelData);
//        // ..do something to the data...
//
//        if (torchInferenceCompleted && torchThread.outputCopied) {
//            buffer.copyFrom(channel, 0, inferenceBuffer, channel, curr_sample, samplesThisTime);
//        } else if (sampleClipLoaded) {
//            buffer.copyFrom(channel, 0, fileBuffer, channel, curr_sample, samplesThisTime);
//        }
//    }
//
//    curr_sample = (curr_sample + samplesThisTime) % sample_size;
   
}

//==============================================================================
bool AudioPluginAudioProcessor::hasEditor() const
{
    return true; // (change this to false if you choose to not supply an editor)
}

juce::AudioProcessorEditor* AudioPluginAudioProcessor::createEditor()
{
    return new AudioPluginAudioProcessorEditor (*this);
}

//==============================================================================
void AudioPluginAudioProcessor::getStateInformation (juce::MemoryBlock& destData)
{
    // You should use this method to store your parameters in the memory block.
    // You could do that either as raw data, or use the XML or ValueTree classes
    // as intermediaries to make it easy to save and load complex data.
    juce::ignoreUnused (destData);
}

void AudioPluginAudioProcessor::setStateInformation (const void* data, int sizeInBytes)
{
    // You should use this method to restore your parameters from this memory block,
    // whose contents will have been created by the getStateInformation() call.
    juce::ignoreUnused (data, sizeInBytes);
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
 
void AudioPluginAudioProcessor::setMorph (bool morph) {
    jassert(torchThread.outputCopied);
    
    auto *morphVoice = dynamic_cast<MorphVoice*>(synth.getVoice(0));
    morphVoice->addMorphed = morph;
}

void AudioPluginAudioProcessor::setMix (float mix) {
//    jassert(torchThread.outputCopied);
    
    auto *morphVoice = dynamic_cast<MorphVoice*>(synth.getVoice(0));
    morphVoice->mixParam = mix;
}


void AudioPluginAudioProcessor::addListener (Listener* newListener) { listeners.add (newListener); }

void AudioPluginAudioProcessor::checkInferenceCompleted () {
    
    
    if (torchThread.inferenceCompleted != torchInferenceCompleted) {
        DBG("check inference");
        torchInferenceCompleted = torchThread.inferenceCompleted;
        torchInferenceInProgress = false;
//        buttonListeners.callChecked (checker, [this] (Listener& l) { l.buttonClicked (this); });
        listeners.call([this] (Listener& l) { l.inferenceStatusChanged (this); });
//        listeners.call(&AudioPluginAudioProcessor::inferenceStatusChanged, this);
    }
    if (torchThread.outputCopied != torchOutputCopied ) {
        DBG("check output copied");
        torchOutputCopied = torchThread.outputCopied;
        listeners.call([this] (Listener& l) { l.inferenceStatusChanged (this); });
    }
    if (torchInferenceCompleted && !torchThread.outputCopied) {
        auto *morphVoice = dynamic_cast<MorphVoice*>(synth.getVoice(0));
        torchThread.copyOutput (&(morphVoice->inferenceBuffer));
    }
    
}

//==============================================================================
// This creates new instances of the plugin..
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new AudioPluginAudioProcessor();
}
