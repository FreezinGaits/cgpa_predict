const express = require('express');
const cors = require('cors');
const multer = require('multer');
const { createObjectCsvWriter } = require('csv-writer');
const path = require('path');
const fs = require('fs');

const app = express();
app.use(cors());
app.use(express.json());

// Setup folders
const UPLOAD_DIR = path.join(__dirname, 'uploads');
if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR);
if (!fs.existsSync(path.join(UPLOAD_DIR, 'audio'))) fs.mkdirSync(path.join(UPLOAD_DIR, 'audio'));
if (!fs.existsSync(path.join(UPLOAD_DIR, 'notes'))) fs.mkdirSync(path.join(UPLOAD_DIR, 'notes'));

// Setup CSVWriter
const csvPath = path.join(__dirname, 'testing_data.csv');
const csvHeader = [
  { id: 'timestamp', title: 'Timestamp' },
  { id: 'email', title: 'Email Address' },
  { id: 'name', title: 'Name' },
  { id: 'roll_number', title: 'University Roll Number' },
  { id: 'university', title: 'University/College' },
  { id: 'prev_gpa', title: 'Previous_Semester_GPA' },
  { id: 'prev_prev_gpa', title: 'CGPA of last to last Semester' },
  { id: 'midterm', title: 'Midterm_Score_Average' },
  { id: 'assignment', title: 'Assignment_Score_Average' },
  { id: 'twelfth', title: 'Twelfth_Grade_Percentage' },
  { id: 'tenth', title: 'Tenth_Grade_Percentage' },
  { id: 'study_hours', title: 'Study_Hours_Per_Day' },
  { id: 'attendance', title: 'Attendance_Percentage' },
  { id: 'backlogs', title: 'Number_of_Backlogs' },
  { id: 'stress', title: 'Mental_Stress_Score' },
  { id: 'distance', title: 'Distance_From_Campus_KM' },
  { id: 'complexity', title: 'Complexity of Content' },
  { id: 'teacher_fb', title: "Teacher's Feedback On Your Presentations" },
  { id: 'participation', title: "Teacher's Feedback On Your Participation" },
  { id: 'photo', title: 'Photo filename' },
  { id: 'audio', title: 'Audio filename' }
];

const csvWriter = createObjectCsvWriter({
  path: csvPath,
  header: csvHeader,
  append: fs.existsSync(csvPath)
});

// Setup File Uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    if (file.fieldname === 'audio') cb(null, path.join(UPLOAD_DIR, 'audio'));
    else if (file.fieldname === 'photo') cb(null, path.join(UPLOAD_DIR, 'notes'));
  },
  filename: (req, file, cb) => {
    cb(null, Date.now() + '-' + file.originalname.replace(/\s+/g, '_'));
  }
});
const upload = multer({ storage });

// POST endpoint
app.post('/submit', upload.fields([{ name: 'audio', maxCount: 1 }, { name: 'photo', maxCount: 1 }]), async (req, res) => {
  try {
    const data = JSON.parse(req.body.data);
    const audioFile = req.files['audio'] ? req.files['audio'][0].filename : '';
    const photoFile = req.files['photo'] ? req.files['photo'][0].filename : '';
    
    await csvWriter.writeRecords([{
      timestamp: new Date().toISOString(),
      ...data,
      audio: audioFile,
      photo: photoFile
    }]);
    
    res.status(200).json({ success: true, message: 'Data saved successfully!' });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: 'Failed to save data' });
  }
});

app.listen(3001, () => {
  console.log('✅ Form backend running on http://localhost:3001');
});
