from fastapi import FastAPI

from routers.user_matching import matchrouter
# from routers.task_extraction import taskrouter

app = FastAPI()
# app.include_router(sleeprouter.calculate_sleep_quality, prefix="/sleep")

# Include routes from the sleep router
# app.include_router(
#     sleeprouter.router,
#     prefix="/ubts"
# )

app.include_router(
    matchrouter.router,
    prefix="/usermatching"
)
# app.include_router(
#     taskrouter.router,
#     prefix='/ubts'
# )