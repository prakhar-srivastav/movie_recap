# movie_recap

It utilizes APIs like TTS (Text-to-Speech), HBAPI, and OpenAI's completion API for generating summaries. MoviePy is employed for video editing, ensuring efficient consolidation of movie clips, while Clip is used for semantic search to enhance the summarization process. This combination of tools and APIs enables the creation of detailed and engaging movie summaries. It has many audit feature as well. You can explore more on the channel [here](https://www.youtube.com/channel/UC7moZjrHEEtrv8AFGmAnokw).'



Test Case 1
QuotesGPTMiddleware

1. synthesize -> working, filepath consistency ✅
2. object addition, save_file ✅
3. get_context, get_internal_context ✅
4. silence addition ✅
5. get subtitles in segment level ✅
__________________________________________

Test Case 2
Function Add-ons

0. regenerate_by_ids ✅
1. whisperx ✅
2. word level ts ✅
3. delete speech [I] ✅
4. get subtites at word level ✅

_____________________________________________

Test Case 3
UI Tests

1. delete function from UI ✅
2. movie and quotes UI flow 

_____________________________________________

Test Case 4
0. video file optimization ✅
1. preset_{i}.py impl. 

_____________________________________________

Test Case 5
Crawler(SM) Tests 
1. movie and quotes crawler ✅ 
