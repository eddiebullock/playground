def exam_pass(scores):
    #count passing scores (>= 60)
    passing_scores = sum(1 for score in scores if score >= 60)

    #calculate percentage 
    total_students = len(scores)
    percentage = (passing_scores / total_students) * 100

    return percentage


raw_data = [14, 14, 14, 14, 41, 100]
result = exam_pass(raw_data)
print(f"{result:.1f} % of students passed")
