IF (${Rcs_RcsGraphics_FOUND})
  IF (NOT TARGET RcsViewerComponent)
    
    SET(SRC ViewerComponent.cpp)
    
    ADD_LIBRARY(RcsViewerComponent ${LIBRARY_LINKING_TYPE} ${SRC})
    TARGET_LINK_LIBRARIES(RcsViewerComponent RcsGraphics)
    TARGET_INCLUDE_DIRECTORIES(RcsViewerComponent PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  
  ENDIF (NOT TARGET RcsViewerComponent)
ENDIF (${Rcs_RcsGraphics_FOUND})
