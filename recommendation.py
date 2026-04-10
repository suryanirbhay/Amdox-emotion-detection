def recommend_task(emotion):
    if emotion == "happy":
        return "Assign creative or collaborative team task."
    elif emotion == "sad":
        return "Assign light task and check employee well-being."
    elif emotion == "angry":
        return "Assign independent work to avoid conflicts."
    elif emotion == "stressed":
        return "Suggest short break and notify HR if repeated."
    else:
        return "Assign regular routine task."
