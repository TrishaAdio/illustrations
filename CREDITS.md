# Credits and licensing of reference material

Read this before using any plate in a published book. The licence of a reference
photograph **propagates to the plate drawn from it**, because a plate is a
derivative work.

## Reference photographs used

| file | source | photographer | licence |
|---|---|---|---|
| `samples/tagore.jpg` | [Rabindranath Tagore, 1909](https://commons.wikimedia.org/wiki/File:Rabindranath_Tagore_in_1909.jpg) | unknown | Public domain (by age) |
| `samples/sarat.jpg` | [Sarat Chandra Chattopadhyay](https://commons.wikimedia.org/wiki/File:Sarat_Chandra_Chattopadhyay.jpg) | unknown | Public domain (by age) |
| `samples/akhilesh-sitter.jpg` | [Rural Farmer in Tamil Nadu](https://commons.wikimedia.org/wiki/File:India_-_Faces_-_Rural_Farmer_in_Tamil_Nadu_(5182306956).jpg) | McKay Savage | **CC BY 2.0** |
| `samples/sitter-evenlight.jpg` | [Delhi old man](https://commons.wikimedia.org/wiki/File:India_-_Delhi_old_man_-_5089.jpg) | Jorge Royan | **CC BY-SA 3.0** |

## What this means in practice

**Public domain** — free to use, no obligations. Both archival portraits qualify
by age.

**CC BY 2.0** (`akhilesh-sitter.jpg`) — usable commercially, requires crediting
the photographer. Safe for a book.

**CC BY-SA 3.0** (`sitter-evenlight.jpg`) — usable commercially, requires credit
**and share-alike**: any plate derived from it must itself be released under
CC BY-SA. That is almost certainly unacceptable for a commercial novel, since it
would mean giving the illustrations away under a copyleft licence.

So `sitter-evenlight.jpg` and everything in `plates/studies/` derived from it are
**technical studies only** — proof that the pipeline works, not artwork for
publication. It was chosen because it has the even lighting the pipeline needs,
not because it is clearable.

## For the actual books

Photograph your own sitter. It resolves licensing completely, and it is also the
only way to get a consistent Akhilesh across thirty plates, since the same face
in the same costume can be re-shot in any pose the story needs.

## Casting and photography spec

Learned the hard way — reference selection dominates every parameter in the tool.
Two sitters were tried; the difference between an unusable plate and a good one
was entirely the photograph.

**Essential:**

1. **Even, soft, diffuse light.** Overcast daylight, or open shade, or a window
   with a curtain. This is the single most important requirement.
2. **No dappled light.** Leaf shadows across a face are fatal. They are
   mid-frequency tonal features, indistinguishable from real structure, so no
   amount of illumination correction removes them. The first sitter failed on
   this alone and produced camouflage.
3. **Plain, uncluttered background**, ideally a light wall, distinctly lighter or
   darker than the subject. This makes background removal work and keeps the
   hatching from turning into noise.
4. **Subject fills the frame.** Head and shoulders, not a full figure in a room.
5. **1500 px on the short edge or better.**

**Avoid:** harsh backlight, direct midday sun, heavy shadow across half the face,
busy backgrounds, and low-resolution or heavily compressed scans.

**For Akhilesh specifically:** an elderly man, short and heavy-shouldered, bald
across the crown with coarse white hair at the sides, short white moustache and
no beard, round steel-rimmed spectacles, a dhoti with a dark coat, and a black
umbrella used as a stick. Shoot three-quarter profile and straight profile, plus
one looking down at a pocket watch, and one seated. Those four cover most of the
plates a book needs.
