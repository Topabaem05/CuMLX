# PyPI 릴리스

`cumlx-lab`을 PyPI에 올리는 절차입니다. 계정과 공개 결정이 필요한 단계는 저장소 소유자만 할 수 있습니다. 나머지는 준비되어 있습니다.

## 준비된 것

- `.github/workflows/release.yml`: 태그 `v<버전>`을 push하면 배포 파일을 만들고, `twine check --strict`와 저장소 밖에서의 wheel 설치 시험(`scripts/wheel_smoke.py`)을 거친 뒤 PyPI에 올립니다. 태그와 `pyproject.toml`의 버전이 다르면 멈춥니다.
- 올리기는 PyPI의 trusted publishing으로 합니다. 저장소에 토큰을 두지 않습니다. PyPI에 publisher를 등록하지 않으면 마지막 단계가 실패하고 아무것도 올라가지 않습니다.
- 2026-11-07에 이 Mac에서 `python -m build`와 `twine check --strict`가 sdist와 wheel 모두 통과했습니다. wheel에는 `cumlx.compat`의 어댑터와 `__fdividef` 표가 들어 있습니다. CI의 `Build distributions and isolated wheel smoke test` job이 커밋마다 같은 검사를 합니다.
- 2026-11-07에 PyPI에서 `cumlx-lab`이라는 이름은 비어 있었습니다(`https://pypi.org/pypi/cumlx-lab/json`이 404).

## 소유자가 할 일

1. **버전을 정합니다.** 지금 `pyproject.toml`은 `0.5.0`이고, `CHANGELOG.md`의 `Unreleased`에 그 뒤의 변경이 쌓여 있습니다. 로드맵은 M2부터 M6까지 `v0.6.0`~`v1.0.0`을 붙였지만 버전을 올리지 않았고, M6-1이 체크되지 않았으므로 `v1.0.0`은 맞지 않습니다. 정한 버전으로 `pyproject.toml`의 `version`과 `CHANGELOG.md`의 `Unreleased` 제목을 바꿔 커밋합니다.
2. **PyPI에 trusted publisher를 등록합니다.** PyPI 계정의 Publishing 화면에서 "pending publisher"를 추가합니다. 프로젝트 이름 `cumlx-lab`, owner `Topabaem05`, repository `CuMLX`, workflow `release.yml`, environment `pypi`.
3. **GitHub에 environment `pypi`를 만듭니다.** 저장소 Settings → Environments. 승인자를 본인으로 두면 올리기 전에 한 번 더 확인할 수 있습니다.
4. **태그를 push합니다.**

   ```bash
   git tag v0.6.0
   ```

   ```bash
   git push origin v0.6.0
   ```

   `v0.6.0`은 예입니다. 1번에서 정한 버전을 씁니다.
5. **확인합니다.** 새 가상환경에서 `pip install 'cumlx-lab[capture,diff]'`로 설치하고 `cumlx selftest`를 실행합니다. 통과하면 [1개월 로드맵](ROADMAP-1MONTH.md)의 "PyPI 게시"를 체크합니다.

## 하지 않는 일

- 태그 없이 올리지 않습니다. 수동 실행(`workflow_dispatch`)은 두지 않았습니다.
- TestPyPI는 쓰지 않습니다. 필요하면 같은 방식으로 publisher를 등록하고 `repository-url`을 더합니다.
