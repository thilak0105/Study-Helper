from django.urls import path
from . import views

urlpatterns = [
    path('', views.dummy, name='dummy'),
    path('upload_study_material/', views.upload_study_material, name='upload_study_material'),
    path('home/', views.home, name='home'),
    path('lessons/<int:study_material_id>/', views.lessons_page, name='lessons_page'),
    path('login/', views.custom_login, name='custom_login'),
    path('signup/', views.signup, name='signup'),
    path('streaks/', views.StreaksAPIView, name='streaks'),
    path('process_study_material/<int:study_material_id>/', views.process_study_material, name='process_study_material'),
    path('profile/', views.profile, name='profile'),
    path('generate_notes/<int:study_material_id>/', views.generate_notes, name='generate_notes'),
    path('start_text_to_speech/', views.start_text_to_speech, name='start_text_to_speech'),
    path('stop_text_to_speech/', views.stop_text_to_speech, name='stop_text_to_speech'),
    path('submit_answers/', views.submit_answers, name='submit_answers'),
    path('translate_pdf/', views.translate_pdf_view, name='translate_pdf'),
    path('questions_form/', views.questions_form, name='questions_form'),  # Ensure this line is present
]