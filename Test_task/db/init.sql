CREATE TABLE IF NOT EXISTS grades (
    id SERIAL PRIMARY KEY,
    full_name VARCHAR(255) NOT NULL,
    subject VARCHAR(255) NOT NULL,
    grade INTEGER NOT NULL CHECK (grade >= 1 AND grade <= 5),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_full_name ON grades(full_name);
CREATE INDEX IF NOT EXISTS idx_grade ON grades(grade);