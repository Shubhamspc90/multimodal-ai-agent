class PcmRecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.buffer = new Float32Array(4096);
    this.offset = 0;

    this.port.onmessage = (event) => {
      if (event.data?.type !== "flush") {
        return;
      }

      if (this.offset > 0) {
        this.port.postMessage(
          this.buffer.slice(0, this.offset),
        );
      }

      this.offset = 0;
      this.port.postMessage({ type: "flushed" });
    };
  }

  process(inputs, outputs) {
    const inputChannels = inputs[0];
    const outputChannels = outputs[0];

    if (outputChannels) {
      for (const channel of outputChannels) {
        channel.fill(0);
      }
    }

    if (!inputChannels || inputChannels.length === 0) {
      return true;
    }

    const frameCount = inputChannels[0].length;

    for (let i = 0; i < frameCount; i += 1) {
      let sample = 0;

      for (const channel of inputChannels) {
        sample += channel[i];
      }

      sample /= inputChannels.length;

      this.buffer[this.offset] = sample;
      this.offset += 1;

      if (this.offset === this.buffer.length) {
        this.port.postMessage(this.buffer.slice());
        this.offset = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-recorder-processor", PcmRecorderProcessor);