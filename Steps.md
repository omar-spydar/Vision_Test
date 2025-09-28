Step 1. Angular size of the tag θ
For a tag of physical width 𝑊 at distance 𝑑
the angular size 𝜃 in radians is given by 𝜃 = 2 arctan(𝑊 / 2𝑑)

Step 2. Pixels per degree px/deg

From camera specs: width in pixels w / HFOV (horizontal field of view in degrees)  

Step 3. Tag width in pixels

tag_px_width=θ × px/deg

Step 4. Tag area in pixels

Since the tag is square: tag_px_area = tag_px_width²

Step 5. Pixels per cell (for decoding reliability)

If the tag dictionary is 𝑛 × 𝑛 bits, the total grid (including 1-cell black border) is:
N=n+2cells across

pixels_per_cell = tag_px_width / N

