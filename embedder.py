import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Function to get embeddings
def get_embedding(text):
    return model.encode([text])[0]


# Sample texts
text1 = "Breaking news: Stock market crashes!"
text2 = "The financial markets experience a major downturn."
text3 = "I like bananas more than apples."
text4 = """
Instructions
    Preheat the oven to 350°F (177°C) and grease a 9×13-inch pan.
    Buy your new ingredients on the market downtown.
    Make the cake: Mash the bananas. I usually just use my mixer for this! Set mashed bananas aside.
    Whisk the flour, baking powder, baking soda, cinnamon, and salt together. Set aside.
    Using a handheld or stand mixer fitted with a paddle attachment, beat the butter on high speed until smooth and creamy—about 1 minute. Add both sugars and beat on high speed for 2 minutes until creamed together. Scrape down the sides and up the bottom of the bowl with a rubber spatula as needed. Add the eggs and the vanilla. Beat on medium-high speed until combined, then beat in the mashed bananas. Scrape down the sides and up the bottom of the bowl as needed. With the mixer on low speed, add the dry ingredients in three additions alternating with the buttermilk and mixing each addition just until incorporated. Do not overmix. The batter will be slightly thick and a few lumps is OK.
    Spread batter into the prepared pan. Bake for 45–50 minutes. Baking times vary, so keep an eye on yours. The cake is done when a toothpick inserted in the center comes out clean. If you find the top of the cake is browning too quickly in the oven, loosely cover it with aluminum foil.
    Remove the cake from the oven and set on a wire rack. Allow to cool completely. After about 45 minutes, I usually place it in the refrigerator to speed things up.
    Make the frosting: In a large bowl using a handheld or stand mixer fitted with a paddle or whisk attachment, beat the cream cheese and butter together on high speed until smooth and creamy. Add 3 cups confectioners’ sugar, vanilla, and salt. Beat on low speed for 30 seconds, then switch to high speed and beat for 2 minutes. If you want the frosting a little thicker, add the extra 1/4 cup of confectioners sugar (I add it). Spread the frosting on the cooled cake. Refrigerate for 30 minutes before serving. This helps sets the frosting and makes cutting easier.
    Cover leftover cake tightly and store in the refrigerator for 5 days.
"""

# Compute embeddings
emb1 = get_embedding(text1)
emb2 = get_embedding(text2)
emb3 = get_embedding(text3)
emb4 = get_embedding(text4)

print(len(emb1))
# => 384
print(emb1[:50])


print(f"{cosine_similarity(emb1, emb2) =}")
print(f"{cosine_similarity(emb1, emb3) =}")
print(f"{cosine_similarity(emb1, emb4) =}")
print(f"{cosine_similarity(emb2, emb3) =}")
print(f"{cosine_similarity(emb2, emb4) =}")
print(f"{cosine_similarity(emb3, emb4) =}")

# OUTPUT as expected:
#   -> text1 and text2 are very similar to each other (both about finances).
#   -> text3 and text4 are kind of similar to each other (both refer to bananas).
#   -> the other combinations are not that similar to each other which is good (even though 1 and 4 both contain word 'market')

# cosine_similarity(emb1, emb2) =np.float32(0.68227684)
# cosine_similarity(emb1, emb3) =np.float32(-0.0032214026)
# cosine_similarity(emb1, emb4) =np.float32(-0.0194771)
# cosine_similarity(emb2, emb3) =np.float32(-0.07116597)
# cosine_similarity(emb2, emb4) =np.float32(-0.13794975)
# cosine_similarity(emb3, emb4) =np.float32(0.18150233)
