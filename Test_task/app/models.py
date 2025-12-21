from pydantic import BaseModel, field_validator

class StudentGradeRow(BaseModel):
    full_name: str
    subject: str
    grade: int

    @field_validator('grade')
    def validate_grade(cls, v):
        if not (1 <= v <= 5):
            raise ValueError('Оценка должна быть от 1 до 5')
        return v
    
    @field_validator('full_name')
    def validate_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Имя не может быть пустым')
        return v

class UploadResponse(BaseModel):
    status: str
    records_loaded: int
    students: int

class StudentStat(BaseModel):
    full_name: str
    count_twos: int