import os
import math
from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import StudyMaterial, Module, CustomUser
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.conf import settings
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from django.utils import timezone
import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect
import pyttsx3
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
from googletrans import Translator
import nest_asyncio
#from torch.cuda.amp import autocast, GradScaler

tokenizer = T5Tokenizer.from_pretrained("t5-base")
# model = T5ForConditionalGeneration.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
#scaler = GradScaler()  

from django.shortcuts import render, redirect
from django.http import HttpResponse
from .models import StudyMaterial, CustomUser, Profile
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.views.decorators.csrf import csrf_protect
from django.conf import settings
import json
import pyttsx3
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

def profile(request):
    """
    Display user profile with user information and study materials.
    """
    user = request.user
    study_materials = StudyMaterial.objects.filter(user=user)
    
    # Handle the case where the Profile object does not exist
    try:
        profile = Profile.objects.get(user=user)
    except Profile.DoesNotExist:
        # Create a new Profile object if it does not exist
        profile = Profile.objects.create(user=user)
    
    context = {
        'user': user,
        'profile': profile,
        'study_materials': study_materials
    }
    
    return render(request, 'profile.html', context)

from django.shortcuts import render, redirect
from django.http import JsonResponse
from .models import StudyMaterial, Evaluation
from django.views.decorators.csrf import csrf_protect

def process_pdf_to_modules(study_material):
    """
    Process uploaded PDF and create modules.
    """
    try:
        # Get the PDF file path
        file_path = study_material.file.path
        
        # Open and read PDF
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        
        # Update total pages in study material
        study_material.total_pages = total_pages
        study_material.save()

        # Calculate pages per module (let's say 10 pages per module)
        pages_per_module = 10
        total_modules = math.ceil(total_pages / pages_per_module)

        # Create modules
        for module_number in range(total_modules):
            start_page = module_number * pages_per_module
            end_page = min((module_number + 1) * pages_per_module - 1, total_pages - 1)
            
            # Extract text from pages in this module
            module_content = ""
            for page_num in range(start_page, end_page + 1):
                page = reader.pages[page_num]
                module_content += page.extract_text() + "\n"

            # Create module
            Module.objects.create(
                study_material=study_material,
                title=f"Module {module_number + 1}",
                content=module_content,
                sequence_number=module_number + 1,
                start_page=start_page + 1,  # Adding 1 for human-readable page numbers
                end_page=end_page + 1,
                estimated_time=15 * (end_page - start_page + 1)  # Estimate 15 minutes per page
            )

        # Mark study material as processed
        study_material.is_processed = True
        study_material.save()

        return True, "PDF processed successfully"

    except Exception as e:
        return False, str(e)

# Debugging: Upload Study Material
@csrf_protect
def upload_study_material(request):
    if request.method == "POST" and request.FILES.get('file'):
        uploaded_file = request.FILES['file']
        user = request.user  # Get the logged-in user
        
        # Debugging: Check file and user info
        print(f"File uploaded: {uploaded_file.name}")
        print(f"User: {user.username}")
        
        # Create a StudyMaterial object and save the file
        try:
            study_material = StudyMaterial(
                user=user,
                file=uploaded_file,
                title=uploaded_file.name,  # Use the uploaded file name as the title
                description=request.POST.get('description', ''),
                total_pages=0,  # Placeholder, you can later add logic to extract this
                is_processed=False  # Set to False by default
            )
            study_material.save()
            print(f"Study material saved with ID: {study_material.id}")
            return redirect('process_study_material', study_material_id=study_material.id)
        except Exception as e:
            # Debugging: Log any exception
            print(f"Error saving study material: {str(e)}")
            return HttpResponse(f"Error uploading file: {str(e)}")

    return HttpResponse("No file uploaded or invalid request method.")

# Debugging: Home page rendering
def home(request):
    print("Rendering home page")
    return render(request, 'home.html')

# Debugging: Lessons Page
def lessons_page(request, study_material_id):
    try:
        study_material = StudyMaterial.objects.get(id=study_material_id)
        modules = Module.objects.filter(study_material=study_material).order_by('sequence_number')

        context = {
            'study_material': study_material,
            'modules': modules,
            
        }

        print(f"Rendering lessons for study material ID: {study_material.id}")
        return render(request, 'lessons.html', context)
    except StudyMaterial.DoesNotExist:
        print(f"Study material with ID {study_material_id} does not exist.")
        return HttpResponse("Study material not found.", status=404)

# Debugging: Custom Login
@csrf_protect
def custom_login(request):
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()

        # Debugging: Login attempt
        print(f"Login attempt for username: {username}")

        # Try to authenticate the user
        user = authenticate(request, username=username, password=password)
        print(f"Authentication result: {user}")

        if user is not None:
            login(request, user)
            messages.success(request, "Login successful!")
            print(f"User {username} logged in successfully.")
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password.")
            print(f"Failed login attempt for username: {username}")
            return redirect('custom_login')

    return render(request, 'login.html')

# Debugging: User Signup
@csrf_protect
def signup(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        email = request.POST.get('email')

        # Debugging: Form data
        print(f"Signup attempt for username: {username}, email: {email}")

        if not username or not password or not email:
            messages.error(request, "All fields are required")
            return redirect('signup')

        if CustomUser.objects.filter(username=username).exists():
            messages.error(request, "Username already exists")
            return redirect('signup')

        try:
            # Create user with create_user which properly hashes the password
            user = CustomUser.objects.create_user(
                username=username,
                email=email,
                password=password  # create_user will hash this automatically
            )
            print(f"User created successfully: {username}")
            messages.success(request, "Account created successfully! Please login.")
            return redirect('custom_login')
        except Exception as e:
            # Debugging: Log any exception during user creation
            print(f"Error creating user: {str(e)}")
            messages.error(request, f"An error occurred: {str(e)}")
            return redirect('signup')

    return render(request, 'signup.html')

# Debugging: Dummy page rendering
def dummy(request):
    print("Rendering dummy page")
    return render(request, 'dashboard.html')

def summarize_text(texts):
    """
    Summarize the extracted texts using the T5 model.
    """
    max_input_length = 1024
    inputs = tokenizer(texts, return_tensors="pt", max_length=max_input_length, truncation=True, padding=True)
    
    # Adjusted parameters for a longer summary
    summary_ids = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_length=500,  # Reduced max length for a more concise summary
        min_length=150,  # Adjusted min length
        length_penalty=1.0,  # Reduced length penalty to allow longer summaries
        num_beams=4, 
        early_stopping=False  # Allow the model to generate up to max_length
    )
    summaries = [tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]
    return summaries


def clean_and_transform_summary(summary):
    """
    Clean and transform the summary into paragraphs or points.
    """
    # Remove unnecessary periods and redundancy
    cleaned_summary = summary.replace('..', '.').replace('. .', '.').strip()
    
    # Split the summary into sentences
    sentences = cleaned_summary.split('. ')
    
    # Remove empty sentences, duplicates, and sentences with only one word or just a number
    sentences = list(filter(None, sentences))
    sentences = list(dict.fromkeys(sentences))
    sentences = [sentence for sentence in sentences if len(sentence.split()) > 1 and not sentence.isdigit()]
    
    # Transform into paragraphs or points
    paragraphs = "\n\n".join(sentences)
    points = "\n".join([f"- {sentence.strip()}" for sentence in sentences])
    
    return paragraphs, points

def rephrase_for_accessibility(summary):
    """
    Rephrase the summary to make it more accessible for people with ADHD and learning disabilities.
    """
    # Split the summary into sentences
    sentences = summary.split('. ')
    
    # Rephrase each sentence to use simpler language and shorter sentences
    rephrased_sentences = []
    for sentence in sentences:
        # Example rephrasing (this can be improved with more sophisticated NLP techniques)
        rephrased_sentence = sentence.replace('however', 'but').replace('therefore', 'so')
        rephrased_sentences.append(rephrased_sentence)
    
    # Convert to bullet points
    bullet_points = "\n".join([f"- {sentence.strip()}" for sentence in rephrased_sentences])
    
    return bullet_points

def process_study_material(request, study_material_id):
    """
    Process the study material: Extract text from the PDF and summarize.
    """
    study_material = StudyMaterial.objects.get(id=study_material_id)
    
    # Extract text from PDF
    file_path = study_material.file.path
    text = extract_text_from_pdf(file_path)

    # Split text into chunks if it exceeds the maximum input length
    max_input_length = 1024
    chunks = [text[i:i + max_input_length] for i in range(0, len(text), max_input_length)]

    # Summarize each chunk in parallel and combine the summaries
    summaries = summarize_text(chunks)
    summary = " ".join(summaries)  # Join the list of summaries into a single string
    
    # Clean and transform the summary
    paragraphs, points = clean_and_transform_summary(summary)
    
    # Split points into lists
    points_list = points.split('\n')
    
    # Save the summary in the StudyMaterial model
    study_material.summary = summary
    study_material.is_processed = True
    study_material.save()

    return render(request, "summary.html", {
        "study_material": study_material,
        "paragraphs": paragraphs,
        "points": points_list
    })

def extract_text_from_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)



import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_protect

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)

@csrf_protect
def start_text_to_speech(request):
    if request.method == "POST":
        try:
            # Get the summary from the POST request body
            data = json.loads(request.body)
            summary = data.get('summary')  # Extract the 'summary' from the request

            # Debugging: Log the summary
            print(f"Received summary: {summary}")

            if summary:
                # Stop any ongoing speech
                engine.stop()
                # Say the new summary
                engine.say(summary)
                engine.runAndWait()
                return JsonResponse({"message": "Speech is being read."}, status=200)
            else:
                return JsonResponse({"error": "No summary provided."}, status=400)

        except Exception as e:
            # Log the error
            print(f"Error: {str(e)}")
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    return JsonResponse({"error": "Invalid request method."}, status=405)

@csrf_protect
def stop_text_to_speech(request):
    if request.method == "POST":
        try:
            # Stop any ongoing speech
            engine.stop()
            return JsonResponse({"message": "Speech has been stopped."}, status=200)
        except Exception as e:
            # Log the error
            print(f"Error: {str(e)}")
            return JsonResponse({"error": f"An error occurred: {str(e)}"}, status=500)
    return JsonResponse({"error": "Invalid request method."}, status=405)

def generate_notes(request, study_material_id):
    """
    Generate notes from the summarized study material.
    """
    study_material = StudyMaterial.objects.get(id=study_material_id)
    
    # Split the summary into bullet points and remove empty points
    points = [point.strip() for point in study_material.summary.split('.') if point.strip()]
    
    return render(request, "notes.html", {
        "study_material": study_material,
        "points": points
    })

def calculate_streaks(profile):
    """
    Calculate the streaks for the user's profile.
    """
    # Example logic for calculating streaks
    current_streak = profile.current_streak
    longest_streak = profile.longest_streak
    last_active_date = profile.last_active_date

    # Update streaks based on the current date
    today = timezone.now().date()
    if last_active_date == today - timezone.timedelta(days=1):
        current_streak += 1
    else:
        current_streak = 1

    if current_streak > longest_streak:
        longest_streak = current_streak

    # Update the profile
    profile.current_streak = current_streak
    profile.longest_streak = longest_streak
    profile.last_active_date = today
    profile.save()

    return {
        "current_streak": current_streak,
        "longest_streak": longest_streak
    }

def StreaksAPIView(request):
    """
    API endpoint to get the streaks of the user.
    """
    user = request.user
    if user.is_authenticated:
        # Get the user's profile
        profile = Profile.objects.get(user=user)
        
        # Calculate the streaks
        streaks = calculate_streaks(profile)
        
        return JsonResponse({"streaks": streaks}, status=200)
    else:
        return JsonResponse({"error": "User not authenticated."}, status=401)
from django.shortcuts import render
from django.http import HttpResponse
from PyPDF2 import PdfReader
import spacy
import random
import re
from django.contrib import messages

# Load spaCy's language model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file):
    """Extract text from an uploaded PDF file."""
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    text = re.sub(r"\s+", " ", text).strip()  # Clean up whitespace
    return text

def generate_questions_with_spacy(text):
    """Generate a list of questions using spaCy's NLP capabilities."""
    doc = nlp(text)
    questions = set()
    used_entities = set()

    question_endings = [
        "important in this context?",
        "relevant for this topic?",
        "significant for understanding the subject?",
        "critical for the overall system?",
        "vital to the discussion?",
    ]

    # Named Entity Recognition-based questions
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PERSON", "PRODUCT"]:
            if ent.text not in used_entities:
                questions.add(f"What is {ent.text}?")
                used_entities.add(ent.text)

    # Verb-based question
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            questions.add(f"How does {token.text} affect the system?")
            break

    # Noun chunk-based questions
    for chunk in doc.noun_chunks:
        ending = random.choice(question_endings)
        questions.add(f"How is {chunk.text} {ending}")

    return list(questions)[:5]  # Return the first 5 questions

def questions_form(request):
    """Render the questions page."""
    if request.method == "POST":
        # Get the generated notes from the POST request body
        notes = request.POST.getlist("notes")
        if notes:
            # Combine notes into a single string
            combined_notes = " ".join(notes)
            # Generate questions from the combined notes
            questions = generate_questions_with_spacy(combined_notes)
            return render(request, "question_form.html", {"questions": questions})
        else:
            messages.error(request, "No notes provided.")
            return render(request, "notes.html", {"error": "No notes provided."})
    return render(request, "question_form.html", {"questions": []})

def submit_answers(request):
    """Handle submission of answers."""
    if request.method == "POST":
        # Get the submitted answers
        questions = request.POST.getlist("questions")
        answers = request.POST.getlist("answers")

        # Pair questions with answers
        qa_pairs = zip(questions, answers)

        # Add a success message
        messages.success(request, "Submission successful!")

        return render(request, "submitted_answers.html", {"qa_pairs": qa_pairs})

    return HttpResponse("Invalid request method.")

import os
import asyncio
import PyPDF2
from googletrans import Translator
from azure.core.credentials import AzureKeyCredential
import nest_asyncio
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_protect

nest_asyncio.apply()

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

async def translate_text_with_azure(text, target_languages=['ta', 'ml', 'te', 'kn', 'hi', 'gu', 'bn']):
    translator = Translator()
    translated_texts = {}
    for lang in target_languages:
        try:
            chunk_size = 15000
            text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            translated_chunks = []
            for chunk in text_chunks:
                translated_chunk = await translator.translate(chunk, dest=lang)
                translated_chunks.append(translated_chunk.text)
            translated_texts[lang] = "".join(translated_chunks)
        except Exception as e:
            print(f"Error translating to {lang}: {e}")
    return translated_texts

@csrf_protect
def translate_pdf_view(request):
    if request.method == "POST":
        pdf_path = request.FILES['pdf_file'].temporary_file_path()
        pdf_text = extract_text_from_pdf(pdf_path)
        if pdf_text:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            translated_texts = loop.run_until_complete(
                translate_text_with_azure(pdf_text, target_languages=['ta'])
            )
            return render(request, "translated_texts.html", {"translated_texts": translated_texts})
        else:
            return HttpResponse("Failed to extract text from the PDF.")
    return HttpResponse("Invalid request method.")