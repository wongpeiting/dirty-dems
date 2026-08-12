"""
Pydantic schema for the DESCRIPTION LAYER (Stage 1).

An "operationalized semantic representation" of what a TikTok communicates —
evidence-bearing, NOT a free-form summary, and NOT taxonomy labels (no sentiment /
mechanism / portrayal; those are Stage 2). Each inference carries the cue that
grounds it, so a human can audit every claim.

Used as the structured output type for both model arms (Gemini native video,
Claude+frames) via pydantic-ai.
"""
from typing import Literal
from pydantic import BaseModel, Field


class CastMember(BaseModel):
    name: str = Field(description="Person or org as identified, e.g. 'Donald Trump', 'Democratic Party'. Use a real name when identifiable; else a description like 'unnamed reporter'.")
    kind: Literal["person", "org", "unknown"]
    how_identified: Literal["face", "named_in_caption", "named_in_audio", "on_screen_text", "voice", "implied"] = Field(description="The strongest cue by which you identified them.")
    role_in_post: Literal["speaking", "spoken_about", "both", "shown_only"]
    is_main: bool = Field(description="True only if the post is centrally about this person/org (the protagonist or the target).")
    evidence: str = Field(description="The concrete cue grounding this entry (what was shown/said/written).")


class OnScreenText(BaseModel):
    text: str = Field(description="Verbatim on-screen text, INCLUDING any emojis exactly as they appear (🤭😭💀🔥).")
    role: Literal["caption", "overlay", "chyron", "meme_text", "credit", "other"]


class VideoDescription(BaseModel):
    video_id: str

    # what is communicated, grounded
    visual_action: str = Field(description="Neutral description of what happens on screen across the clip — people, setting, actions, B-roll, screenshots. No interpretation of intent yet.")
    cast: list[CastMember] = Field(description="Every person/org named, shown, or clearly implied. Empty list ONLY if genuinely no person/org appears.")
    on_screen_text: list[OnScreenText] = Field(description="All readable on-screen text, with emojis preserved.")
    spoken_summary: str = Field(description="What is said (audio/dialogue), reconciled with the provided transcript. Quote distinctive lines.")
    emojis_present: list[str] = Field(default_factory=list, description="Unicode emojis that appear on screen or in the caption (deduplicated).")

    # craft signals (descriptive, not yet coded)
    audio_music: str = Field(description="Describe the audio's STRUCTURE OVER TIME and its ROLE/effect — e.g. 'silent for the first ~5s, then an ominous score enters', or 'upbeat track that switches to a hip-hop beat over the montage'. Note silence, when sound starts, and any track CHANGES. Describe effect (ominous, hype, patriotic, mocking). Do NOT assert a specific song title/artist as fact — song identity comes from a separate music-ID service; if you think you recognize a track you may note it as 'sounds like…' but never state it as certain.")
    editing_pacing: str = Field(description="Editing style — fast cuts, speed ramps, zooms, reaction shots, slow-mo, static talking-head, slideshow.")
    is_manipulated: bool = Field(default=False, description="True if the post contains deliberate VISUAL fabrication/manipulation you can SEE: photoshop, a superimposed object (e.g. a sombrero added onto a politician), deepfake, AI-generated imagery, or a staged fake document/broadcast — made for mockery or effect. NOT true merely because a date is unfamiliar or an event is after your knowledge cutoff.")
    manipulation_or_satire: str = Field(default="", description="If is_manipulated, describe exactly what is fabricated/altered and what it's doing — e.g. 'a sombrero is digitally superimposed on Schumer and Jeffries to mock them as pro-illegal-immigration'. Empty string if nothing is manipulated.")

    # communicative reading, grounded
    emotions_signaled: list[str] = Field(description="The emotions the post is engineered to evoke (e.g. 'mockery', 'fear', 'pride'), each tied to a cue where possible.")
    cultural_references: list[str] = Field(default_factory=list, description="Pop-culture BORROWING — a key attention tactic; capture it FULLY, never omit. Include: memes, movie/TV/game/song references, internet formats & trends, AND famous NON-political people or fictional characters invoked or shown (actors, musicians, athletes, cartoon characters) — e.g. a Brad Pitt / Jim Carrey reaction clip, a Hannah Montana (Miley Cyrus) meme, a Sabrina Carpenter SNL skit, SpongeBob. ALSO note famous people genuinely appearing/visiting (e.g. Cristiano Ronaldo or a sports team at the White House). Name each person/character specifically. (Political figures go in `cast`, not here.)")
    central_claim: str = Field(description="The post's core communicative claim or implication, in one sentence — what it is really saying.")
    intended_effect: str = Field(description="The reaction it seems engineered to produce in the viewer (e.g. 'laugh at Vance', 'fear the opponent', 'feel proud').")

    # audit
    uncertainty_flags: list[str] = Field(default_factory=list, description="Anything you are unsure about or could not determine — drives human review.")
