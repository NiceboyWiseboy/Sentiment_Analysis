from models.schemas import InputText
from routers.senti import process_text
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse


router = APIRouter(tags=['apis'])


@router.post('/input_text')
async def analyze_text(input_text: InputText):
    try:
        output = process_text(input_text=input_text.text)
        print(output)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                'success': True,
                'result': output
            }
        )
    except Exception as error:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                'success': False,
                'result': error
            }
        )
