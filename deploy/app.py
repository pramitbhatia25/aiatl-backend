from flask import Flask, redirect, render_template, request, url_for, jsonify
app = Flask(__name__)
import vertexai
from vertexai.language_models import TextGenerationModel
from vertexai.vision_models import ImageTextModel, Image
import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

model_options = ['Cat', 'Guitar', 'Airplane', 'human', 'Bird']

@app.route("/", methods=("GET", "POST"))
def home():
    return render_template("page.html", title="HOME PAGE")
    

@app.route("/backend", methods=("GET", "POST"))
def backend():
    if request.method == "POST":
        r.set("Chats", "")
        try:
            # Generate Caption From Image
            image_file = request.files.get("file")
            image_content = image_file.read()
            # Save the image to a file if needed
            with open("image.jpg", "wb") as f:
                f.write(image_content)

            captions = get_caption()
            print(captions)

            # Generate Personality from caption and prompt
            if captions:
                try:
                    prompt = request.headers.get("prompt", "")
                    response = generate_personality(prompt, captions)
                    personality = response.text

                    if personality:
                        try:
                            response = classify(captions, model_options)
                            return jsonify({"personality": personality, "model": response.text})

                        except Exception as e:
                            print(e)
                            print("Couldn't classify.")

                    # Return the captions and personality as JSON
                    return jsonify({"personality": personality, "model": "human"})

                except Exception as e:
                    print(e)
                    print("Couldn't create personality.")

        except Exception as e:
            print("No image file provided in the request")

    return jsonify({"error": "Invalid request"})


def get_caption():
    vertexai.init(project='aiatl-405516', location='us-central1')
    model = ImageTextModel.from_pretrained("imagetext@001")

    source_image = Image.load_from_file(location='./image.jpg')

    captions = model.get_captions(
        image=source_image,
        number_of_results=1,
        language="en",
    )
    return captions


def generate_prompt(prompt, captions):
    return (f"""Hello there! I'm a personality creator,
     and I'm here to give life to an object. The following is a description of the object:
    {captions}. The personality of the object should be STRICTLY based on the following:
    {prompt}. Imagine that object is not just a name but
    a unique character with its own quirks and traits.
    Tell me, what kind of personality do you envision for
    that object? Is it a mischievous puppy that
    loves to play pranks on humans, or perhaps a wise
    old guy that imparts knowledge to anyone who opens his pages?
    Let your imagination run wild, and let's bring the object to life!
    Make sure the personality suits the description of what the object is and the prompt.""")


def generate_personality(prompt, captions, temperature: float = 0.2) -> None:
    vertexai.init(project='aiatl-405516', location='us-central1')
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        generate_prompt(prompt, captions),
        **parameters,
    )
    return response


def classify(captions, model_options, temperature: float = 0.2):
    vertexai.init(project='aiatl-405516', location='us-central1')
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        f"""Given the description '{captions}', classify the object into one of {model_options}.
        Ensure the model returns only the name of the best-matching object with 100% certainty,
        excluding any additional text. The best matching model MUST BE CHOSEN from the options given to you for classification.
        DO NOT CHOOSE A RANDOM MODEL NAME. If none of the options match, return 'Sphere' as the answer.""",
        **parameters,
    )
    return response


@app.route("/chatbot", methods=("GET", "POST"))
def chatbot():
    message = request.headers["message"]
    personality = request.headers["personality"]
    
    
    try:
        current_chats = r.get("Chats")
        r.append("Chats", f"USER: {message}\n")
        total_chats = r.get("Chats")
        response = chat(total_chats, personality)
        r.append("Chats", f"{response}\n")
        total_chats = r.get("Chats")
        print("A" + str(total_chats))
    except Exception as e:
        total_chats = r.set("Chats", f"USER: {message}\n")
        response = chat(total_chats, personality)
        r.append("Chats", f"{response}\n")
        total_chats = r.get("Chats")
        print("B" + str(total_chats))

    return jsonify({"response": response})


def chat(message, personality, temperature: float = 0.2):
    vertexai.init(project='aiatl-405516', location='us-central1')
    parameters = {
        "temperature": temperature,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
        "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        f"""You are a chatbot! Your personality is described as such: {personality}.
        Completely go into the character described by your personality.
        Assume that you ARE the character described in your personality.
        Think about how that character might reply. Maybe use famous quotes that they've used.
        Repond in the best way possible to the message.
        ALSO, keep in mind your percious chats that have existed with this person. Here's a record of the previous chats:{message}""",
        **parameters,
    )
    print("ABRACADABRA")
    print(response.text)
    print("ABRACADABRA")
    return response.text


if __name__ == "__main__":
    app.run()