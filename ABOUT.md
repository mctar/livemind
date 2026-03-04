# Live Mind Map

## What is it?

Live Mind Map turns a conversation into a visual map of ideas — in real time.

As people talk in a meeting, workshop, or lecture, the system listens, transcribes what's being said, and uses AI to identify the key concepts and how they connect. These appear as an animated, interactive map on screen: glowing nodes for ideas, lines for relationships, colours for categories.

The map grows and evolves as the conversation unfolds. New ideas appear. Old ones fade if the discussion moves on. The result is a living picture of what a group is actually talking about — not what someone planned to talk about.

## Who is it for?

- **Facilitators and workshop leaders** who want a real-time visual of group discussion
- **Meeting hosts** who need to capture the shape of a conversation, not just minutes
- **Teachers and lecturers** who want to show students how ideas connect
- **Conference speakers** who want interactive, audience-driven visuals
- **Anyone** curious about what a conversation looks like when you can see it

## How does it work?

1. **You speak.** The system captures audio from a microphone using local speech-to-text — nothing leaves your machine for transcription.
2. **AI reads the transcript.** Every 20 seconds, the new text is sent to Claude (Anthropic's AI) which identifies the important concepts and their relationships.
3. **The map updates.** New concepts appear with a gentle animation. Connections draw themselves. The map rebalances to stay readable.
4. **You interact.** Right-click any concept to pin it, rename it, merge duplicates, or hide noise. The map is yours to shape.

## What makes it different?

- **Real-time.** Not a summary after the fact. The map builds as people talk.
- **Local speech processing.** Audio never leaves your computer. Transcription runs on-device using Apple Silicon.
- **Smart lifecycle.** The system doesn't just add things — it manages what stays visible. Ideas that stop being discussed gradually fade. Important ideas stick around. The map stays clean even in a 2-hour session.
- **Session memory.** Everything is saved. You can close the browser, come back, and pick up where you left off. Start a new session when you're ready.
- **No accounts, no cloud dependency.** Runs on your laptop. The only external call is to the Claude API for concept extraction.

## What does it look like?

A dark canvas with softly glowing nodes connected by curved lines. Each concept category gets its own colour. New ideas pop in with a brief animation. The whole thing breathes — subtle movements keep it feeling alive without being distracting.

On the right side, a scrolling transcript shows what's being said, with the most recent text highlighted.

## What do you need to run it?

- A Mac with Apple Silicon (M1 or later) — for local speech-to-text
- An Anthropic API key — for the AI concept extraction
- Python 3.11+ and a modern browser
- A microphone

That's it. No frameworks to install, no Docker, no database servers. One command starts the whole thing.
