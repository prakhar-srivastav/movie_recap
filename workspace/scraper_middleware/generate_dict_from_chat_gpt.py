import traceback
import json

def generate_dict_from_chat_gpt(system, content, mandatory_keys, model_type = 'gpt-4-turbo-preview', retries = 3):
        from openai import OpenAI
        client = OpenAI(api_key = 'sk-proj-L8a41e1mHEpmUjFK4wc0T3BlbkFJ7v9ugFqkIL01Hatfst3T')
        print('Fetching data from gpt api for system', system)
        for i in range(retries):
            print('try count: {},'.format(i+1))
            try:
                response = client.chat.completions.create(
                                model= model_type,
                                messages=[
                                    {
                                    "role": "system",
                                    "content": system
                                    },
                                    {
                                    "role": "user",
                                    "content": content
                                    },],
                                temperature=1,
                                max_tokens=4095,
                                top_p=1,
                                frequency_penalty=0,
                                presence_penalty=0
                                )
                
                message = response.dict()['choices'][0]['message']['content']
                message = message.replace('\n',' ')
                message = json.loads(message)
                for v in mandatory_keys:
                    assert v in message
                print('Done Fetching')
                return message
            except:
                print(traceback.format_exc())
                pass
        raise ValueError('Fetch Failed')