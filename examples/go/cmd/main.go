package main

import (
	"log"
	"os"

	"github.com/streamer45/silero-vad-go/speech"

	"github.com/go-audio/wav"
)

func main() {
	sd, err := speech.NewDetector(speech.DetectorConfig{
		ModelPath:            "../../files/silero_vad.onnx",
		SampleRate:           16000,
		WindowSize:           1536,
		Threshold:            0.5,
		MinSilenceDurationMs: 0,
		SpeechPadMs:          0,
	})
	if err != nil {
		log.Fatalf("failed to create speech detector: %s", err)
	}

	f, err := os.Open(os.Args[1])
	if err != nil {
		log.Fatalf("failed to open sample audio file: %s", err)
	}
	defer f.Close()

	dec := wav.NewDecoder(f)

	if ok := dec.IsValidFile(); !ok {
		log.Fatalf("invalid WAV file")
	}

	buf, err := dec.FullPCMBuffer()
	if err != nil {
		log.Fatalf("failed to get PCM buffer")
	}

	pcmBuf := buf.AsFloat32Buffer()

	segments, err := sd.Detect(pcmBuf.Data)
	if err != nil {
		log.Fatalf("Detect failed: %s", err)
	}

	for _, s := range segments {
		log.Printf("speech starts at %0.2fs", s.SpeechStartAt)
		if s.SpeechEndAt > 0 {
			log.Printf("speech ends at %0.2fs", s.SpeechEndAt)
		}
	}

	err = sd.Destroy()
	if err != nil {
		log.Fatalf("failed to destroy detector: %s", err)
	}
}
