import csv
import io
import os
from contextlib import asynccontextmanager
from typing import List

import asyncpg
from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse

from app.models import StudentGradeRow, UploadResponse, StudentStat

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/students_db")

db_pool = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    print("Создание пула подключений к БД...")
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    yield
    print("Закрытие пула подключений...")
    await db_pool.close()

app = FastAPI(title="Students Grades Service", lifespan=lifespan)

@app.post("/upload-grades", response_model=UploadResponse)
async def upload_grades(file: UploadFile):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате .csv")

    content = await file.read()
    decoded_content = content.decode('utf-8')
    
    csv_reader = csv.DictReader(io.StringIO(decoded_content), delimiter=',')
    
    valid_records = []
    unique_students = set()
    
    for row in csv_reader:
        if not {'full_name', 'subject', 'grade'}.issubset(row.keys()):
             raise HTTPException(status_code=400, detail="CSV должен содержать заголовки: full_name, subject, grade")
        
        try:
            item = StudentGradeRow(
                full_name=row['full_name'],
                subject=row['subject'],
                grade=int(row['grade'])
            )
            valid_records.append((item.full_name, item.subject, item.grade))
            unique_students.add(item.full_name)
        except (ValueError, KeyError) as e:
            continue

    if not valid_records:
        raise HTTPException(status_code=400, detail="В файле нет валидных данных")

    query = """
        INSERT INTO grades (full_name, subject, grade) 
        VALUES ($1, $2, $3)
    """
    
    async with db_pool.acquire() as connection:
        async with connection.transaction():
            await connection.executemany(query, valid_records)

    return {
        "status": "ok",
        "records_loaded": len(valid_records),
        "students": len(unique_students)
    }

@app.get("/students/more-than-3-twos", response_model=List[StudentStat])
async def get_students_many_twos():
    query = """
        SELECT full_name, COUNT(*) as count_twos
        FROM grades
        WHERE grade = 2
        GROUP BY full_name
        HAVING COUNT(*) > 3
        ORDER BY count_twos DESC;
    """
    
    async with db_pool.acquire() as connection:
        rows = await connection.fetch(query)
    
    return [dict(row) for row in rows]

@app.get("/students/less-than-5-twos", response_model=List[StudentStat])
async def get_students_few_twos():
    """
    Возвращает студентов, у которых двоек меньше 5.
    Важный нюанс: сюда входят и те, у кого 0 двоек, но есть другие оценки.
    Поэтому используем CASE WHEN внутри SUM/COUNT, а не WHERE grade=2.
    """
    query = """
        SELECT full_name, 
               SUM(CASE WHEN grade = 2 THEN 1 ELSE 0 END) as count_twos
        FROM grades
        GROUP BY full_name
        HAVING SUM(CASE WHEN grade = 2 THEN 1 ELSE 0 END) < 5
        ORDER BY count_twos ASC;
    """
    
    async with db_pool.acquire() as connection:
        rows = await connection.fetch(query)
    
    return [dict(row) for row in rows]