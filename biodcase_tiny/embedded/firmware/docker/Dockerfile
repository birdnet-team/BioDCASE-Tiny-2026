ARG IDF_TAG
FROM docker.io/espressif/idf:$IDF_TAG
ARG UNAME="nonroot"
ARG UID="1000"
ARG GID="1000"

# Add non-root user.
RUN echo "$GID"
RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -G $UNAME -o -s /bin/bash $UNAME
RUN echo '. /opt/esp/idf/export.sh' >> "/home/$UNAME/.bashrc"
RUN chown -R $UNAME /opt/esp/idf/

# grant dialout and plugdev to user
RUN usermod -a -G dialout $UNAME
RUN usermod -a -G plugdev $UNAME

USER $UNAME
